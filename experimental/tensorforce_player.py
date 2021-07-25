from catanatron.state_functions import get_player_actual_vps
import random
import os
import shutil
from pathlib import Path

from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorforce import Agent, Environment
import click

from catanatron.game import Game
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron_gym.features import (
    create_sample_vector,
    get_feature_ordering,
)
from catanatron_gym.envs.catanatron_env import (
    ACTIONS_ARRAY,
    ACTION_SPACE_SIZE,
    normalize_action,
)
from experimental.machine_learning.players.minimax import (
    AlphaBetaPlayer,
    ValueFunctionPlayer,
)

# For repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

FEATURES = get_feature_ordering(2)
NUM_FEATURES = len(FEATURES)
EPISODES = 10_000  # 25_000 is like 8 hours
EPISODES = 25_000

EPISODES_PER_PHASE = 3_000
PHASES = 10


@click.command()
@click.argument("experiment_name")
def main(experiment_name):
    # Pre-defined or custom environment
    # environment = Environment.create(
    #     environment="gym", level="CartPole", max_episode_timesteps=500
    # )
    environment = Environment.create(
        environment=CustomEnvironment, max_episode_timesteps=1000
    )

    # initialize enemy and agent.
    checkpoint_dir = Path("data", "checkpoints", experiment_name)
    enemy_checkpoint_dir = Path("data", "checkpoints", f"{experiment_name}-enemy")
    logs_dir = Path("data", "logs", experiment_name)
    if checkpoint_dir.exists():
        # try loading latest enemy or default to random
        if enemy_checkpoint_dir.exists():
            print("Loading enemy...")
            environment._environment.enemy_agent = ForcePlayer(
                Color.RED, str(Path(enemy_checkpoint_dir, "enemy"))
            )

        print("Loading agent...")
        agent = Agent.load(directory=str(checkpoint_dir), environment=environment)
    else:
        agent = create_agent(environment, logs_dir, checkpoint_dir)

    # start phase, play games, store at phase 1 path.
    for _ in tqdm(range(1, PHASES + 1), ascii=True, unit="phase"):
        for _ in tqdm(range(1, EPISODES_PER_PHASE + 1), ascii=True, unit="episodes"):
            # Initialize episode
            states = environment.reset()
            terminal = False

            while not terminal:
                # Episode timestep
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)

        # done with phase, so copy phase over to enemy.
        print("Saving to", enemy_checkpoint_dir, "enemy", "saved-model")
        agent.save(enemy_checkpoint_dir, "enemy", "saved-model")
        # print("Copying from", checkpoint_dir, enemy_checkpoint_dir)
        # shutil.copytree(checkpoint_dir, enemy_checkpoint_dir, dirs_exist_ok=True)

        # Reload enemy
        environment._environment.enemy_agent = ForcePlayer(
            Color.RED, str(Path(enemy_checkpoint_dir, "enemy"))
        )

    agent.close()
    environment.close()


def create_agent(environment, log_dir, checkpoint_dir):
    print("Creating model...")
    return Agent.create(
        agent="vpg",
        environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
        memory=50_000,  # alphazero is 500,000
        batch_size=32,
        update_frequency=1,
        exploration=dict(
            type="linear",
            unit="episodes",
            num_steps=EPISODES,
            initial_value=1.0,
            final_value=0.05,
        ),
        l2_regularization=1e-4,
        summarizer=dict(
            directory=str(log_dir),
            summaries=["reward", "action-value", "parameters"],
        ),
        saver=dict(
            directory=str(checkpoint_dir),
            frequency=100,  # save checkpoint every N updates
        ),
    )


class CustomEnvironment(Environment):
    enemy_agent = RandomPlayer(Color.RED)
    # RandomPlayer(Color.RED),
    # VictoryPointPlayer(Color.RED),
    # ValueFunctionPlayer(Color.RED),

    def states(self):
        return dict(type="float", shape=(NUM_FEATURES,))

    def actions(self):
        return dict(type="int", num_values=ACTION_SPACE_SIZE)

    def reset(self):
        p0 = Player(Color.BLUE)
        players = [p0, self.enemy_agent]
        game = Game(players=players)
        self.game = game
        self.p0 = p0

        self._advance_until_p0_decision()
        return build_states(self.game, self.p0)

    def execute(self, actions):
        action = from_action_space(actions, self.game.state.playable_actions)
        self.game.execute(action)
        self._advance_until_p0_decision()

        winning_color = self.game.winning_color()
        next_state = build_states(self.game, self.p0)
        terminal = winning_color is not None

        # Implement a not-so-sparse reward that rewards vps over opponent
        vps = get_player_actual_vps(self.game.state, self.p0.color)
        enemy_color = next(filter(lambda c: c != self.p0.color, self.game.state.colors))
        enemy_vps = get_player_actual_vps(self.game.state, enemy_color)
        reward = 0
        if self.p0.color == winning_color:
            reward += 0.75
        elif winning_color is not None:  # enemy won
            reward += -0.75
        vp_diff = vps - enemy_vps  # pos if we have more
        reward += vp_diff * 0.02

        return next_state, terminal, reward

    def _advance_until_p0_decision(self):
        while (
            self.game.winning_color() is None
            and self.game.state.current_player().color != self.p0.color
        ):
            self.game.play_tick()  # will play bot


def build_states(game, p0):
    sample = create_sample_vector(game, p0.color)

    action_ints = list(map(to_action_space, game.state.playable_actions))
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    mask[action_ints] = True

    states = dict(state=sample, action_mask=mask)
    return states


MODEL = None


class ForcePlayer(Player):
    def __init__(self, color, model_path):
        super(ForcePlayer, self).__init__(color)
        global MODEL
        # MODEL = Agent.load(directory=checkpoints_dir)
        # MODEL.spec["summarizer"] = None
        # MODEL.spec["saver"] = None
        MODEL = tf.keras.models.load_model(model_path)

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        states = build_states(game, self)

        # Create Tensor(1,614) and mask to Tensor(1,290)
        tstater = tf.reshape(tf.convert_to_tensor(states["state"]), (1, 614))
        tmaskr = tf.reshape(tf.convert_to_tensor(states["action_mask"]), (1, 290))
        result = MODEL.act(tstater, {"mask": tmaskr}, tf.convert_to_tensor(True))
        action_int = result.numpy()[0]

        # action_int = MODEL.act(states, independent=True)

        best_action = from_action_space(action_int, playable_actions)
        return best_action


def to_action_space(action):
    """maps action to space_action equivalent integer"""
    normalized = normalize_action(action)
    return ACTIONS_ARRAY.index((normalized.action_type, normalized.value))


def from_action_space(action_int, playable_actions):
    """maps action_int to catantron.models.actions.Action"""
    # Get "catan_action" based on space action.
    # i.e. Take first action in playable that matches ACTIONS_ARRAY blueprint
    (action_type, value) = ACTIONS_ARRAY[action_int]
    catan_action = None
    for action in playable_actions:
        normalized = normalize_action(action)
        if normalized.action_type == action_type and normalized.value == value:
            catan_action = action
            break  # return the first one
    assert catan_action is not None
    return catan_action


if __name__ == "__main__":
    main()

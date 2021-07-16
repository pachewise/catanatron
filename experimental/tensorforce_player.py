import random
import os
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

EPISODES_PER_PHASE = 1000
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
    base_checkpoint_dir = Path("data", "checkpoints", experiment_name)
    base_enemy_checkpoint_dir = Path("data", "checkpoints", f"{experiment_name}-enemy")
    base_logs_dir = Path("data", "logs", experiment_name)

    # has runned before
    has_runned_before = base_checkpoint_dir.exists()
    if has_runned_before:
        print("Continuing where we left...")

        # try loading latest checkpoint, or create
        phases = sorted([int(x) for x in os.listdir(base_checkpoint_dir)])
        if len(phases) == 0:
            latest_phase = 1
            log_dir = Path(base_logs_dir, "1")
            latest_checkpoint_dir = Path(base_checkpoint_dir, "1")
            agent = create_agent(environment, log_dir, latest_checkpoint_dir)
        else:
            latest_phase = phases[-1]
            latest_checkpoint_dir = Path(base_checkpoint_dir, str(latest_phase))
            agent = Agent.load(directory=str(latest_checkpoint_dir))
            print("Loaded agent", latest_phase)

        # try loading latest enemy or default to random
        if base_enemy_checkpoint_dir.exists():
            enemy_phases = sorted(
                [int(x) for x in os.listdir(base_enemy_checkpoint_dir)]
            )
            if len(enemy_phases) > 0:
                latest_enemy_phase = enemy_phases[-1]
                latest_enemy_checkpoint_dir = Path(
                    base_enemy_checkpoint_dir, str(latest_enemy_phase)
                )
                environment._environment.enemy_agent = ForcePlayer(
                    Color.RED, latest_enemy_checkpoint_dir
                )
                print("Loaded enemy", latest_enemy_phase)
    else:
        latest_phase = 1
        log_dir = Path(base_logs_dir, "1")
        latest_checkpoint_dir = Path(base_checkpoint_dir, "1")
        agent = create_agent(environment, log_dir, latest_checkpoint_dir)

    breakpoint()

    # start phase, play games, store at phase 1 path.
    for phase_i in tqdm(range(latest_phase, PHASES + 1), ascii=True, unit="phase"):
        for _ in tqdm(range(1, EPISODES_PER_PHASE + 1), ascii=True, unit="episodes"):
            # Initialize episode
            states = environment.reset()
            terminal = False

            while not terminal:
                # Episode timestep
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)

        # done with phase, so copy phase over to enemy. and maybe set enemy.
        # data/checkpoints/1 => data/checkpoints/2. create_agent(2)
        next_checkponit_dir = Path(base_checkpoint_dir, phase_i + 1)
        print("Copying from", latest_checkpoint_dir, next_checkponit_dir)

        agent.close()
    environment.close()


def create_agent(environment, log_dir, checkpoint_dir):
    print("Creating model...")
    return Agent.create(
        agent="vpg",
        environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
        memory=50_000,  # alphazero is 500,000
        batch_size=32,
        # update=dict(unit="episodes", batch_size=32),
        # optimizer=dict(type="adam", learning_rate=1e-3),
        # policy=dict(network="auto"),
        # exploration=0.05,
        exploration=dict(
            type="linear",
            unit="episodes",
            num_steps=EPISODES,
            initial_value=1.0,
            final_value=0.05,
        ),
        # policy=dict(network=dict(type='layered', layers=[dict(type='dense', size=32)])),
        # objective="policy_gradient",
        # reward_estimation=dict(horizon=20, discount=0.999),
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

        # key = player_key(self.game.state, self.p0.color)
        # points = self.game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
        # reward = int(winning_color == self.p0.color) * 1000 + points
        if self.p0.color == winning_color:
            reward = 1
        elif winning_color is None:
            reward = 0
        else:
            reward = -1
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
    def __init__(self, color, checkpoints_dir):
        super(ForcePlayer, self).__init__(color)
        global MODEL
        MODEL = Agent.load(directory=checkpoints_dir)
        MODEL.spec["summarizer"] = None
        MODEL.spec["saver"] = None

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        states = build_states(game, self)
        action_int = MODEL.act(states, independent=True)
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

import pickle
import random
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from catanatron.models.player import Player
from catanatron.models.actions import ActionType
from machine_learning.features import create_sample

model_path = Path("./catanatron/players/estimator.pickle").resolve()
with open(model_path, "rb") as f:
    model = pickle.load(f)


class GreedyEstimatePlayer(Player):
    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        # print(len(playable_actions))
        my_index = game.players.index(self)

        # For each action, consider playing it.
        samples = []
        for action in playable_actions:
            action_copy = copy.deepcopy(action)
            game_copy = copy.deepcopy(game)
            # print("Trying to execute in copy:", action_copy)
            game_copy.execute(action_copy)

            [p0, p1, p2, p3] = [
                game_copy.players[(i + my_index) % len(game_copy.players)]
                for i in range(len(game_copy.players))
            ]
            sample = create_sample(game_copy, p0, p1, p2, p3)
            samples.append(sample)

        X = pd.DataFrame.from_records(samples)

        scores = model.predict_proba(X)
        best_idx = np.argmax(scores, axis=0)[1]

        # print(X)
        # print(scores)
        # print("Decided", best_idx, scores[best_idx], playable_actions[best_idx])
        return playable_actions[best_idx]


# def road_building_possibilities(player, board):
#     """
#     On purpose we _dont_ remove equivalent possibilities, since we need to be
#     able to handle high branching degree anyway in AI.
#     """
#     first_edges = board.buildable_edges(player.color)
#     possibilities = []
#     for first_edge in first_edges:
#         board_copy = copy.deepcopy(board)
#         first_edge_copy = board_copy.get_edge_by_id(first_edge.id)
#         board_copy.build_road(player.color, first_edge_copy)
#         second_edges_copy = board_copy.buildable_edges(player.color)

#         for second_edge_copy in second_edges_copy:
#             second_edge = board.get_edge_by_id(second_edge_copy.id)
#             possibilities.append(
#                 Action(player, ActionType.PLAY_ROAD_BUILDING, (first_edge, second_edge))
#             )

#     return possibilities

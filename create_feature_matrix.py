import json
import copy
import pandas as pd
import numpy as np

from database import get_finished_games_ids, get_last_game_state, get_game_states
from catanatron.game import Game
from catanatron.json import GameEncoder
from machine_learning.plot import plot_feature_importances, predict
from machine_learning.features import features, create_sample


# Read games from game_states table and create samples.
#   For each state: (p1, p2, p3, p4, winner), (p2, p3, p4, p1, winner), ...
samples = []
labels = []
for game_id in get_finished_games_ids(limit=1000):
    game = get_last_game_state(game_id)
    print(game_id, game)

    players = game.players
    winner = game.winning_player()
    if winner is None:
        print("SKIPPING NOT FINISHED GAME", game)
        continue

    for state in get_game_states(game_id):
        for i, player in enumerate(players):
            p1, p2, p3 = [
                players[(i + 1) % len(players)],
                players[(i + 2) % len(players)],
                players[(i + 3) % len(players)],
            ]
            samples.append(create_sample(game, player, p1, p2, p3))

        for i, player in enumerate(players):
            label = player == winner
            labels.append(label)

print(len(samples))
if len(samples) > 0:
    X = pd.DataFrame.from_records(samples)
    Y = np.array(labels)
    print(X.head())
    print(X.describe())

    print("Predicting...")
    predict(X, Y)
    plot_feature_importances(X, Y)

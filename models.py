import os
from typing import List, Union
from itertools import combinations, permutations

import pandas as pd
import numpy as np


class Deck(object):
    def __init__(self,
                 deck_name: str,
                 csv_data_dir: str = "crawl/output/",
                 win_rate_discount: float = 1.,
                 ):
        self.deck_name = deck_name
        self.csv_data_dir = csv_data_dir
        self.win_rate_discount = win_rate_discount

        self.rank_data = self._load_rank_data()
        self.limit_id = str(self.rank_data['Limit_ID'].iloc[0])
        # print(self.rank_data)
        # print(self.limit_id)
        self.match_data = self._load_match_data()
        # print(self.match_data)

    def _load_rank_data(self, csv_fn: str = "00_rank_data.csv"):
        csv_file = os.path.join(self.csv_data_dir, csv_fn)

        # Read the CSV file into a DataFrame
        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"Error: File {csv_file} not found.")
            return None

        # Filter the DataFrame for the specific deck based on limit_id
        deck_data = df[df['Deck'] == self.deck_name].copy()

        if deck_data.empty:
            print(f"No data found for Deck Name: {self.deck_name}")
            return None

        # Convert data type
        deck_data['Win%'] = deck_data['Win%'].str.strip('%').astype(float)

        return deck_data

    def _load_match_data(self):
        csv_fn = f"{self.limit_id}.csv"
        csv_file = os.path.join(self.csv_data_dir, csv_fn)

        # Read the CSV file into a DataFrame
        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"Error: File {csv_file} not found.")
            return None

        # Process the match data
        df['Win%'] = df['Win%'].str.strip('%').astype(float)  # Convert Win% to float

        return df

    def vs_single(self, opponent: Union[str, "Deck"]) -> float:
        if isinstance(opponent, str):
            win_rate = self.match_data[self.match_data['Opponent'] == opponent]['Win%'].iloc[0]
        elif isinstance(opponent, Deck):
            win_rate = self.match_data[(self.match_data['Opponent'] == opponent.deck_name)]['Win%'].iloc[0]
        else:
            raise TypeError(f"Opponent type {type(opponent)} is not supported.")

        # print(win_rate)
        return float(win_rate) * self.win_rate_discount


class Team(object):
    def __init__(self,
                 team_deck_names: List[str],
                 win_rate_discounts: List[float],
                 ):
        self.team_deck_names = team_deck_names
        self.win_rate_discount = win_rate_discounts

        # Create Deck classes
        self.decks = [Deck(n, win_rate_discount=d)
                      for n, d, in zip(self.team_deck_names, self.win_rate_discount)]

    def vs_team(self, opponents: Union[List[str], "Team"]) -> np.ndarray:
        """ Output a win rate matrix. """
        win_rate_matrix = []

        for one_deck in self.decks:
            deck_win_rates = []

            if isinstance(opponents, list):
                for opponent in opponents:
                    win_rate = one_deck.vs_single(opponent)
                    deck_win_rates.append(win_rate)
            elif isinstance(opponents, Team):
                for opponent in opponents.decks:
                    win_rate = one_deck.vs_single(opponent)
                    deck_win_rates.append(win_rate)
            else:
                raise TypeError(f"Opponents type {type(opponents)} is not supported.")

            win_rate_matrix.append(deck_win_rates)

        return np.array(win_rate_matrix)

    def best_picking_policy(self, opponents: "Team", pick_size: int = 3,
                            opponents_pick_policy: List[tuple] = None,
                            self_banned: int = None,
                            opponent_banned: int = None,
                            ):
        """Find the best picking policy to maximize the average win rate."""
        win_rate_matrix = self.vs_team(opponents)

        # Get all possible combinations of pick_size decks from both teams

        team_combinations = list(combinations(range(len(self.decks)), pick_size))
        if opponents_pick_policy is None:
            opponent_combinations = list(combinations(range(len(opponents.decks)), pick_size))
        else:
            opponent_combinations = opponents_pick_policy

        # Check banned decks
        team_combinations = [d for d in team_combinations if self_banned not in d]
        opponent_combinations = [d for d in opponent_combinations if opponent_banned not in d]
        # print(team_combinations)
        # print(opponent_combinations)

        # Statistics
        best_avg_3x3_win_rate = -np.inf
        best_team_pick_avg_3x3 = None
        best_top_3x2_win_rate = -np.inf
        best_team_pick_top_3x2 = None

        for team_pick in team_combinations:
            # print(f"team_pick: {team_pick}")
            # Assuming the sub-matrix is 3x3
            avg_1x3_win_rates = []  # mean, (M,3)
            avg_3x3_win_rates = []  # mean, (M,1)

            top_1x2_win_rates = []  # top-2, (M,3)
            top_3x2_win_rates = []  # top-2, (M,1)

            for opponent_pick in opponent_combinations:
                # Extract the sub-matrix for the current picks
                sub_matrix = win_rate_matrix[np.ix_(team_pick, opponent_pick)]

                # Obtain statistic data
                avg_1x3_win_rate = sub_matrix.mean(axis=1)
                avg_3x3_win_rate = avg_1x3_win_rate.mean()

                top_2_indices = np.argsort(sub_matrix, axis=1)[:, -2:]
                top_1x2_matrix = np.take_along_axis(sub_matrix, top_2_indices, axis=1)
                top_1x2_win_rate = top_1x2_matrix.mean(axis=1)
                top_3x2_win_rate = top_1x2_win_rate.mean()

                # print(sub_matrix)
                # print(top_1x2_win_rate)
                # print(top_3x2_win_rate)

                avg_1x3_win_rates.append(avg_1x3_win_rate)
                avg_3x3_win_rates.append(avg_3x3_win_rate)
                top_1x2_win_rates.append(top_1x2_win_rate)
                top_3x2_win_rates.append(top_3x2_win_rate)

            avg_1x3_win_rates = np.concatenate(avg_1x3_win_rates, axis=0)
            avg_3x3_win_rates = np.array(avg_3x3_win_rates)
            top_1x2_win_rates = np.concatenate(top_1x2_win_rates, axis=0)
            top_3x2_win_rates = np.array(top_3x2_win_rates)
            # print(avg_3x3_win_rates.mean())
            # print(top_3x2_win_rates.mean())

            # Update the best policy
            if avg_3x3_win_rates.mean() > best_avg_3x3_win_rate:
                best_avg_3x3_win_rate = avg_3x3_win_rates.mean()
                best_team_pick_avg_3x3 = team_pick
            if top_3x2_win_rates.mean() > best_top_3x2_win_rate:
                best_top_3x2_win_rate = top_3x2_win_rates.mean()
                best_team_pick_top_3x2 = team_pick

        return best_avg_3x3_win_rate, best_team_pick_avg_3x3, best_top_3x2_win_rate, best_team_pick_top_3x2

    def best_ban_policy(self, opponents: "Team", ban_size: int = 1,
                            opponents_ban_policy: List[int] = None):
        if ban_size != 1:
            raise NotImplementedError("Ban size should be 1!")

        best_ban_idx = None
        best_ban_avg_wr = -np.inf

        # Get self banned deck
        self_banned = None
        if opponents_ban_policy is not None:
            self_banned = opponents_ban_policy[0]

        # Try different ban targets
        for ban_idx in range(len(opponents.decks)):
            # print(f"Ban Opponent: {ban_idx}")
            avg_wr, avg_pick, top2_wr, top_pick = self.best_picking_policy(
                opponents, opponent_banned=ban_idx,
                self_banned=self_banned,
            )
            # print(avg_wr, top2_wr)
            if avg_wr > best_ban_avg_wr:
                best_ban_avg_wr = avg_wr
                best_ban_idx = ban_idx

        return best_ban_idx, best_ban_avg_wr


# Example usage
# deck = Deck(deck_name="Lugia Archeops")
deck = Deck(deck_name="Mew Genesect")
deck.vs_single(opponent="Lugia Archeops")

team1_decks = ["Lugia Archeops", "Mew Genesect", "Zoroark Box", "Lugia Archeops", "Kyurem Palkia", "Other"]
team1_discounts = [1 for _ in range(len(team1_decks))]
team1_discounts[3] *= 0.95
team2_decks = ["Lugia Archeops", "Mew Genesect", "Arceus Tapu Koko", "Regis", "Goodra LZ Box", "Lost Zone Box"]
team2_discounts = [1] * len(team1_decks)

Team1 = Team(team1_decks, team1_discounts)
Team2 = Team(team2_decks, team2_discounts)
result = Team1.vs_team(opponents=Team2)

# Opponent decides to ban our decks
banning_our_deck, banning_wr = Team2.best_ban_policy(Team1)
print(f"Opponent Will Ban Our Deck: {Team1.decks[banning_our_deck].deck_name}({banning_our_deck}), "
      f"Avg.Win%={banning_wr:.2f}")

# Based on predicted opponent's banning deck, we decide to ban opponent's deck
banning_opponent_deck, banning_wr = Team1.best_ban_policy(Team2)
print(f"We Will Ban Opponent Deck: {Team2.decks[banning_opponent_deck].deck_name}({banning_opponent_deck}), "
      f"Avg.Win%={banning_wr:.2f}")

# Find the best picking policy for opponent
best_avg_wr, best_avg_pick, best_top2_wr, best_top2_pick = Team2.best_picking_policy(
    Team1, pick_size=3, self_banned=banning_opponent_deck)
opponents_picking = [best_avg_pick, best_top2_pick]
print("Opponent Best Average Win Rate:", best_avg_wr, "Top-2 Win Rate:", best_top2_wr)

# Find the best picking policy
opponents_picking = None
best_avg_wr, best_avg_pick, best_top2_wr, best_top2_pick = \
    Team1.best_picking_policy(
        Team2, pick_size=3,
        opponents_pick_policy=opponents_picking,
        self_banned=banning_our_deck,
        opponent_banned=banning_opponent_deck,
    )
print("Best Average Win Rate:", best_avg_wr)
print("Best Average Team Pick:", [team1_decks[i] for i in best_avg_pick])
print("Best Top-2 Win Rate:", best_top2_wr)
print("Best Top-2 Team Pick:", [team1_decks[i] for i in best_top2_pick])

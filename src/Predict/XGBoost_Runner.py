import copy

import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc


# from src.Utils.Dictionaries import team_index_current
# from src.Utils.tools import get_json_data, to_data_frame, get_todays_games_json, create_todays_games
init()
xgb_ml = xgb.Booster()
xgb_ml.load_model('Models/XGBoost_Models/XGBoost_65.45%_ML-4.json')


def xgb_runner(data, games, home_team_odds, away_team_odds, kelly_criterion):
    ml_predictions_array = []

    for row in data:
        ml_predictions_array.append(xgb_ml.predict(xgb.DMatrix(np.array([row])))[0])  # Directly get probability

    count = 0
    for game in games:
        home_team, away_team = game
        home_prob = ml_predictions_array[count]  # Probability of home team winning
        away_prob = 1 - home_prob  # Probability of away team winning

        if home_prob > 0.5:
            winner = home_team
            loser = away_team
            winner_prob = home_prob
            loser_prob = away_prob
        else:
            winner = away_team
            loser = home_team
            winner_prob = away_prob
            loser_prob = home_prob

        # ANSI escape codes for colors
        GREEN = "\033[92m"
        RED = "\033[91m"
        RESET = "\033[0m"

        # Print winner in green with winning percentage and loser in red with losing percentage
        print(
            f"{GREEN}{winner} ({winner_prob * 100:.2f}%){RESET} vs {RED}{loser} ({loser_prob * 100:.2f}%){RESET}")

        count += 1

    # Expected Value & Kelly Criterion Calculation
    if kelly_criterion:
        print("------------Expected Value & Kelly Criterion-----------")
    else:
        print("---------------------Expected Value--------------------")

    count = 0
    for game in games:
        home_team, away_team = game
        home_prob = ml_predictions_array[count]
        away_prob = 1 - home_prob

        ev_home = ev_away = 0
        if home_team_odds[count] and away_team_odds[count]:
            ev_home = round(float(Expected_Value.expected_value(home_prob, int(home_team_odds[count]))), 2)
            ev_away = round(float(Expected_Value.expected_value(away_prob, int(away_team_odds[count]))), 2)

        expected_value_colors = {'home_color': Fore.GREEN if ev_home > 0 else Fore.RED,
                                 'away_color': Fore.GREEN if ev_away > 0 else Fore.RED}

        bankroll_descriptor = ' FoB: '
        bankroll_fraction_home = f"{bankroll_descriptor}{kc.calculate_kelly_criterion(home_team_odds[count], home_prob):.2f}%"
        bankroll_fraction_away = f"{bankroll_descriptor}{kc.calculate_kelly_criterion(away_team_odds[count], away_prob):.2f}%"

        print(home_team + ' EV: ' + expected_value_colors['home_color'] + str(ev_home) + Style.RESET_ALL + (bankroll_fraction_home if kelly_criterion else ''))
        print(away_team + ' EV: ' + expected_value_colors['away_color'] + str(ev_away) + Style.RESET_ALL + (bankroll_fraction_away if kelly_criterion else ''))
        count += 1

    deinit()

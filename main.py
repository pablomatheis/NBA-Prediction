from datetime import datetime, timedelta

import pandas as pd

from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.Predict import XGBoost_Runner
from src.Utils.Dictionaries import team_index_current
from src.Utils.tools import create_todays_games_from_odds, get_json_data, to_data_frame
from polymarket import get_nba_game_odds  # Import Polymarket function
from config import data_url


def get_polymarket_odds(games):
    odds = {}
    for game in games:
        home_team, away_team = game
        polymarket_odds = get_nba_game_odds(home_team, away_team)

        if isinstance(polymarket_odds, dict):
            def get_team_key(team_name):
                words = team_name.split()
                return ' '.join(words[-2:]) if len(words) > 1 else words[0]

            home_team_key = get_team_key(home_team)
            away_team_key = get_team_key(away_team)

            home_odds = polymarket_odds['american_odds'].get(home_team_key, '0')
            away_odds = polymarket_odds['american_odds'].get(away_team_key, '0')

            if home_odds == '0':
                home_team_key = home_team.split()[-1]
                home_odds = polymarket_odds['american_odds'].get(home_team_key, '0')

            if away_odds == '0':
                away_team_key = away_team.split()[-1]
                away_odds = polymarket_odds['american_odds'].get(away_team_key, '0')

            odds[f"{home_team}:{away_team}"] = {
                home_team: {'money_line_odds': home_odds},
                away_team: {'money_line_odds': away_odds},
                'under_over_odds': '0'  # Polymarket doesn't provide under/over odds
            }

            print(f"\033[34m{home_team}\033[0m ({home_odds})({polymarket_odds['polymarket_odds'][home_team_key]}) "
                  f"vs \033[34m{away_team}\033[0m ({away_odds})({polymarket_odds['polymarket_odds'][away_team_key]})")
        else:
            print(f"Couldn't fetch odds for {home_team} vs {away_team} from Polymarket.")

    return odds


def create_todays_games(games, df, odds):
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []

    for game in games:
        home_team, away_team = game

        if home_team not in team_index_current or away_team not in team_index_current:
            continue

        game_key = f"{home_team}:{away_team}"
        if game_key in odds:
            game_odds = odds[game_key]
            todays_games_uo.append(game_odds['under_over_odds'])
            home_team_odds.append(game_odds[home_team]['money_line_odds'])
            away_team_odds.append(game_odds[away_team]['money_line_odds'])
        else:
            todays_games_uo.append('0')
            home_team_odds.append('0')
            away_team_odds.append('0')

        schedule_df = pd.read_csv('Data/nba-2024-UTC.csv', parse_dates=['Date'], date_format='%d/%m/%Y %H:%M')
        home_games = schedule_df[(schedule_df['Home Team'] == home_team) | (schedule_df['Away Team'] == home_team)]
        away_games = schedule_df[(schedule_df['Home Team'] == away_team) | (schedule_df['Away Team'] == away_team)]

        home_days_off = timedelta(days=7)
        away_days_off = timedelta(days=7)

        previous_home_games = \
            home_games.loc[schedule_df['Date'] <= datetime.today()].sort_values('Date', ascending=False).head(1)['Date']
        previous_away_games = \
            away_games.loc[schedule_df['Date'] <= datetime.today()].sort_values('Date', ascending=False).head(1)['Date']

        if not previous_home_games.empty:
            home_days_off = timedelta(days=1) + datetime.today() - previous_home_games.iloc[0]
        if not previous_away_games.empty:
            away_days_off = timedelta(days=1) + datetime.today() - previous_away_games.iloc[0]

        # Get team statistics
        home_team_series = df.iloc[team_index_current.get(home_team)]
        away_team_series = df.iloc[team_index_current.get(away_team)]

        # Calculate winning percentage
        home_wins, home_losses = home_team_series['W'], home_team_series['L']
        away_wins, away_losses = away_team_series['W'], away_team_series['L']

        home_winning_percentage = home_wins / (home_wins + home_losses) if (home_wins + home_losses) > 0 else 0
        away_winning_percentage = away_wins / (away_wins + away_losses) if (away_wins + away_losses) > 0 else 0

        # Add new features
        stats = pd.concat([home_team_series, away_team_series])
        stats['Days-Rest-Home'] = home_days_off.days
        stats['Days-Rest-Away'] = away_days_off.days
        stats['Winning-Percentage-Home'] = home_winning_percentage
        stats['Winning-Percentage-Away'] = away_winning_percentage

        match_data.append(stats)

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1).T
    frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'TEAM_NAME'])
    data = frame_ml.values.astype(float)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds


def main():
    print("Attempting to fetch odds from Polymarket...")

    # Get initial games list from SbrOddsProvider (fanduel)
    initial_odds = SbrOddsProvider(sportsbook='fanduel').get_odds()
    games = create_todays_games_from_odds(initial_odds)

    if not games:
        print("No games found.")
        return

    # Now fetch Polymarket odds for these games
    odds = get_polymarket_odds(games)

    data = get_json_data(data_url)
    df = to_data_frame(data)
    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = create_todays_games(games, df, odds)

    print("---------------XGBoost Model Predictions---------------")
    XGBoost_Runner.xgb_runner(data, games, home_team_odds, away_team_odds, True)  # Always use kc=True
    print("-------------------------------------------------------")


if __name__ == "__main__":
    main()

import argparse
from datetime import datetime, timedelta

import pandas as pd

from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.Predict import XGBoost_Runner
from src.Utils.Dictionaries import team_index_current
from src.Utils.tools import create_todays_games_from_odds, get_json_data, to_data_frame
from polymarket import get_nba_game_odds  # Import Polymarket function

data_url = 'https://stats.nba.com/stats/leaguedashteamstats?' \
           'Conference=&DateFrom=&DateTo=&Division=&GameScope=&' \
           'GameSegment=&LastNGames=0&LeagueID=00&Location=&' \
           'MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&' \
           'PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&' \
           'PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&' \
           'Season=2024-25&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&' \
           'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='


def createTodaysGames(games, df, odds):
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []

    for game in games:
        home_team, away_team = game

        if home_team not in team_index_current or away_team not in team_index_current:
            continue

        if odds and f"{home_team}:{away_team}" in odds:
            game_odds = odds[f"{home_team}:{away_team}"]
            todays_games_uo.append(game_odds['under_over_odds'])
            home_team_odds.append(game_odds[home_team]['money_line_odds'])
            away_team_odds.append(game_odds[away_team]['money_line_odds'])

        else:
            polymarket_odds = get_nba_game_odds(home_team, away_team)

            if isinstance(polymarket_odds, dict):
                def get_team_key(team_name):
                    words = team_name.split()
                    if len(words) > 1:
                        return ' '.join(words[-2:])  # Get last two words for multi-word names
                    return words[0]  # Get the only word for single-word names

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

                todays_games_uo.append('0')  # Polymarket doesn't provide under/over odds
                home_team_odds.append(home_odds)
                away_team_odds.append(away_odds)

                print(f"\033[34m{home_team}\033[0m ({home_odds})({polymarket_odds['polymarket_odds'][home_team_key]}) "
                      f"vs \033[34m{away_team}\033[0m ({away_odds})({polymarket_odds['polymarket_odds'][away_team_key]})")

            else:
                print(f"Couldn't fetch odds for {home_team} vs {away_team} from Polymarket.")
                todays_games_uo.append(0)
                home_team_odds.append(0)
                away_team_odds.append(0)

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

        home_team_series = df.iloc[team_index_current.get(home_team)]
        away_team_series = df.iloc[team_index_current.get(away_team)]
        stats = pd.concat([home_team_series, away_team_series])
        stats['Days-Rest-Home'] = home_days_off.days
        stats['Days-Rest-Away'] = away_days_off.days
        match_data.append(stats)

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1).T
    frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'TEAM_NAME'])
    data = frame_ml.values.astype(float)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds


def main():
    odds = None  # Initialize odds to None

    # Fetch odds from SbrOddsProvider only if -odds argument isn't provided
    if not args.odds:
        print("No -odds argument provided, attempting to fetch odds from Polymarket...")
    else:
        odds_provider = SbrOddsProvider(sportsbook=args.odds)
        odds = odds_provider.get_odds()
        print(f"------------------{args.odds} odds data------------------")
        for g, game_odds in odds.items():
            home_team, away_team = g.split(":")
            print(
                f"{away_team} ({game_odds[away_team]['money_line_odds']}) @ {home_team} ({game_odds[home_team]['money_line_odds']})")

    # Create games list from either sbr or nba API
    if odds:
        games = create_todays_games_from_odds(odds)
    else:
        games = create_todays_games_from_odds(SbrOddsProvider(sportsbook='fanduel').get_odds())

    if not games:
        print("No games found.")
        return

    data = get_json_data(data_url)
    df = to_data_frame(data)
    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = createTodaysGames(games, df, odds)

    print("---------------XGBoost Model Predictions---------------")
    XGBoost_Runner.xgb_runner(data, games, home_team_odds, away_team_odds, args.kc)
    print("-------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XGBoost Model for NBA Predictions')
    parser.add_argument('-odds',
                        help='Sportsbook to fetch from. (fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny')
    parser.add_argument('-kc', action='store_true', help='Calculates percentage of bankroll to bet based on model edge')
    args = parser.parse_args()
    main()


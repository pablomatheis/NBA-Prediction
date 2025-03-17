from pprint import pprint
import requests
import json
from datetime import datetime, timedelta
import pytz


def polymarket_to_american(p):
    probability = p * 100
    if not 0 < probability < 100:
        return "Error: Probability must be between 0 and 100"

    if probability < 50:
        # Positive odds
        odds = (100 / (probability / 100)) - 100
        return f"+{int(round(odds))}"
    else:
        # Negative odds
        odds = (probability / (1 - (probability / 100))) * -1
        return f"{int(round(odds))}"


def fetch_upcoming_nba_games(end_date_max=None, start_date_min=None):
    url = "https://gamma-api.polymarket.com/events"
    params = {
        "tag_id": 745,
        "related_tags": "true",
        "offset": 0,
        "limit": 500,
        "closed": "false",
        'active': 'true'
    }
    if end_date_max:
        params['end_date_max'] = end_date_max
    if start_date_min:
        params['start_date_min'] = start_date_min
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def process_games(events):
    nba_games = []
    for event in events:
        for market in event.get("markets", []):
            try:
                outcomes = json.loads(market.get("outcomes", "[]"))
                best_ask = round(float(market.get("bestAsk", 0)), 2)
                best_bid = round(float(market.get("bestBid", 0)), 2)
                spread = round(float(market.get("spread", 0)), 2)

                if len(outcomes) != 2:
                    continue

                raw_odds = {
                    outcomes[0]: round(best_ask, 2),
                    outcomes[1]: round(1 - best_ask + spread, 2)
                }

                # Calculate the other team's best Bid
                other_team_best_bid = round(1 - best_bid - spread, 2)

                american_odds = {team: polymarket_to_american(prob) for team, prob in raw_odds.items()}
                game_data = {
                    "teams": market.get("question", "N/A"),
                    "odds": raw_odds,
                    "american_odds": american_odds,
                    "best_bid": best_bid,
                    "other_team_best_bid": other_team_best_bid,
                    "url": f"https://polymarket.com/market/{market.get('slug', '')}",
                    "end_date": event.get("endDate", "N/A"),
                    "start_date": event.get("startDate", "N/A")
                }
                nba_games.append(game_data)
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                print(f"Error processing market: {e}")
                continue
    return nba_games


def filter_games_by_date(nba_games, max_end_date):
    return [
        game for game in nba_games
        if game['end_date'] != "N/A" and game['end_date'] <= max_end_date.isoformat()
    ]


def find_game_by_teams(nba_games, t1, t2):
    team1 = t1.split()[-1].lower()
    team2 = t2.split()[-1].lower()
    for game in nba_games:
        teams = game['teams'].lower()
        if team1 in teams and team2 in teams:
            return game
    return None


def get_nba_game_odds(team1, team2):
    try:
        spain_tz = pytz.timezone('Europe/Madrid')
        us_eastern_tz = pytz.timezone('US/Eastern')
        current_time_spain = datetime.now(spain_tz)
        current_time_us = current_time_spain.astimezone(us_eastern_tz)

        max_end_date = current_time_us + timedelta(weeks=1)
        end_date_max = max_end_date.isoformat()

        min_start_date = current_time_us - timedelta(weeks=2)
        start_date_min = min_start_date.isoformat()

        data = fetch_upcoming_nba_games(end_date_max=end_date_max, start_date_min=start_date_min)
        nba_games = process_games(data)
        filtered_nba_games = filter_games_by_date(nba_games, max_end_date)
        game = find_game_by_teams(filtered_nba_games, team1, team2)

        if game:
            return {
                "teams": game['teams'],
                "polymarket_odds": game['odds'],
                "american_odds": game['american_odds'],
                "home_bid": game['best_bid'],
                "away_bid": game['other_team_best_bid'],
                "url": game['url'],
                "end_date": game['end_date'],
                "start_date": game['start_date']
            }
        else:
            return f"No game found between {team1} and {team2} in the upcoming schedule."

    except requests.exceptions.RequestException as e:
        return f"API Error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


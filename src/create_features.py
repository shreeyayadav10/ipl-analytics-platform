import pandas as pd

print("Loading dataset...")

matches = pd.read_csv(
    "data/processed/processed_matches.csv"
)

# ==================================================
# TEAM WIN RATE FEATURE
# ==================================================

team_matches = {}

for _, row in matches.iterrows():

    team1 = row["team1"]
    team2 = row["team2"]

    team_matches[team1] = (
        team_matches.get(team1, 0) + 1
    )

    team_matches[team2] = (
        team_matches.get(team2, 0) + 1
    )

team_wins = matches["winner"].value_counts().to_dict()

team_win_rate = {}

for team in team_matches:

    wins = team_wins.get(team, 0)

    matches_played = team_matches[team]

    team_win_rate[team] = (
        wins / matches_played
    )

matches["team1_win_rate"] = (
    matches["team1"].map(team_win_rate)
)

matches["team2_win_rate"] = (
    matches["team2"].map(team_win_rate)
)

# ==================================================
# VENUE STRENGTH FEATURE
# ==================================================

venue_wins = {}

for _, row in matches.iterrows():

    winning_team = row["winner"]
    venue = row["venue"]

    key = (
        winning_team,
        venue
    )

    venue_wins[key] = (
        venue_wins.get(key, 0) + 1
    )

venue_strength = []

for _, row in matches.iterrows():

    key = (
        row["team1"],
        row["venue"]
    )

    venue_strength.append(
        venue_wins.get(key, 0)
    )

matches["team1_venue_wins"] = (
    venue_strength
)

# ==================================================
# HEAD TO HEAD FEATURE
# ==================================================

head_to_head = {}

for _, row in matches.iterrows():

    team1 = row["team1"]
    team2 = row["team2"]
    winner = row["winner"]

    teams = tuple(
        sorted(
            [team1, team2]
        )
    )

    if teams not in head_to_head:

        head_to_head[teams] = {}

    head_to_head[teams][winner] = (
        head_to_head[teams].get(winner, 0) + 1
    )

head_to_head_advantage = []

for _, row in matches.iterrows():

    team1 = row["team1"]
    team2 = row["team2"]

    teams = tuple(
        sorted(
            [team1, team2]
        )
    )

    records = head_to_head.get(
        teams,
        {}
    )

    team1_wins = records.get(
        team1,
        0
    )

    team2_wins = records.get(
        team2,
        0
    )

    advantage = (
        team1_wins - team2_wins
    )

    head_to_head_advantage.append(
        advantage
    )

matches["head_to_head_advantage"] = (
    head_to_head_advantage
)

# ==================================================
# FEATURE CHECKS
# ==================================================

print("\n===== TEAM WIN RATE FEATURE =====")

print(
    matches[
        [
            "team1",
            "team1_win_rate",
            "team2",
            "team2_win_rate"
        ]
    ].head()
)

print("\n===== VENUE STRENGTH FEATURE =====")

print(
    matches[
        [
            "team1",
            "venue",
            "team1_venue_wins"
        ]
    ].head()
)

print("\n===== HEAD TO HEAD FEATURE =====")

print(
    matches[
        [
            "team1",
            "team2",
            "head_to_head_advantage"
        ]
    ].head()
)

# ==================================================
# SAVE FEATURE DATASET
# ==================================================

matches.to_csv(
    "data/processed/featured_matches.csv",
    index=False
)

print("\nFeature dataset saved successfully.")
print(
    f"Final dataset shape: {matches.shape}"
)
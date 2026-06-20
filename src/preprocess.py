import pandas as pd

print("Reading IPL matches dataset...")

matches = pd.read_csv(
    "data/raw/matches.csv"
)

print(
    f"Original records: {len(matches)}"
)

# =====================================
# TEAM NAME STANDARDIZATION
# =====================================

TEAM_NAME_MAPPING = {

    "Royal Challengers Bangalore":
        "Royal Challengers Bengaluru",

    "Delhi Daredevils":
        "Delhi Capitals",

    "Kings XI Punjab":
        "Punjab Kings",

    "Rising Pune Supergiants":
        "Rising Pune Supergiant"
}

team_columns = [
    "team1",
    "team2",
    "winner",
    "toss_winner"
]

for column in team_columns:

    matches[column] = (
        matches[column]
        .replace(TEAM_NAME_MAPPING)
    )

# =====================================
# ACTIVE IPL TEAMS ONLY
# =====================================

ACTIVE_TEAMS = [

    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Punjab Kings",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Gujarat Titans",
    "Lucknow Super Giants"
]

matches = matches[
    matches["team1"].isin(ACTIVE_TEAMS)
    &
    matches["team2"].isin(ACTIVE_TEAMS)
]

print(
    f"Records after active team filtering: {len(matches)}"
)

# =====================================
# BASIC CLEANING
# =====================================

matches = matches.dropna(
    subset=["winner"]
)

matches["city"] = (
    matches["city"]
    .fillna("Unknown")
)

# =====================================
# SELECT FINAL COLUMNS
# =====================================

final_data = matches[
    [
        "team1",
        "team2",
        "toss_winner",
        "toss_decision",
        "venue",
        "city",
        "winner"
    ]
]

print(
    f"Records after cleaning: {len(final_data)}"
)

unique_teams = sorted(
    set(final_data["team1"])
    .union(
        set(final_data["team2"])
    )
)

print("\nActive Teams Found:")

for team in unique_teams:
    print(team)

print(
    f"\nTotal Active Teams: {len(unique_teams)}"
)

# =====================================
# SAVE DATASET
# =====================================

final_data.to_csv(
    "data/processed/processed_matches.csv",
    index=False
)

print(
    "\nProcessed dataset saved successfully."
)
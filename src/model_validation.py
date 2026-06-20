import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

print("Loading dataset...")

data = pd.read_csv(
    "data/processed/featured_matches.csv"
)

X = data.drop(columns=["winner"])
y = data["winner"]

categorical_features = [
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "venue",
    "city"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)

pipeline = Pipeline(
    [
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42
            )
        )
    ]
)

scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross Validation Scores:")
print(scores)

print(
    f"\nAverage Accuracy: {scores.mean():.4f}"
)
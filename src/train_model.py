import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

print("Loading feature dataset...")

data = pd.read_csv(
    "data/processed/featured_matches.csv"
)

X = data.drop(
    columns=["winner"]
)

y = data["winner"]

categorical_features = [
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "venue",
    "city"
]

numeric_features = [
    "team1_win_rate",
    "team2_win_rate",
    "team1_venue_wins",
    "head_to_head_advantage"
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

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

pipeline = Pipeline(
    [
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training model...")

pipeline.fit(
    X_train,
    y_train
)

predictions = pipeline.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    f"Improved Model Accuracy: {accuracy:.4f}"
)

joblib.dump(
    pipeline,
    "models/model.pkl"
)

print(
    "Updated model saved successfully."
)
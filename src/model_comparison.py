import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

print("Loading dataset...")

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

models = {
    "Logistic Regression":
        LogisticRegression(
            max_iter=2000
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        )
}

results = []

for model_name, model in models.items():

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

    print(f"\nTraining {model_name}...")

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
        f"{model_name}: {accuracy:.4f}"
    )

    results.append(
        {
            "Model": model_name,
            "Accuracy": round(
                accuracy,
                4
            )
        }
    )

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\n========================")
print(results_df)
print("========================")

print(
    "\nResults saved to reports/model_comparison.csv"
)
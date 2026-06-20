import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from sklearn.model_selection import train_test_split

print("Loading dataset...")

data = pd.read_csv(
    "data/processed/featured_matches.csv"
)

X = data.drop(
    columns=["winner"]
)

y = data["winner"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Loading model...")

model = joblib.load(
    "models/model.pkl"
)

predictions = model.predict(
    X_test
)

print("\n===== CLASSIFICATION REPORT =====\n")

print(
    classification_report(
        y_test,
        predictions
    )
)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)

plt.figure(
    figsize=(10, 8)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot(
    xticks_rotation=90
)

plt.tight_layout()

plt.savefig(
    "reports/confusion_matrix.png"
)

plt.show()

print(
    "\nConfusion matrix saved to reports/confusion_matrix.png"
)
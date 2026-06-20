import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("models/model.pkl")

rf_model = model.named_steps["model"]

feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

importance = rf_model.feature_importances_

feature_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importance
    }
)

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

print(feature_df.head(15))

plt.figure(figsize=(10,6))

plt.barh(
    feature_df["Feature"].head(15),
    feature_df["Importance"].head(15)
)

plt.title(
    "Top 15 Important Features"
)

plt.tight_layout()

plt.savefig(
    "reports/feature_importance.png"
)

plt.show()
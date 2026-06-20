import pandas as pd
import joblib

model = joblib.load(
    "models/model.pkl"
)

sample_match = pd.DataFrame(
    {
        "team1": ["Mumbai Indians"],
        "team2": ["Chennai Super Kings"],
        "toss_winner": ["Mumbai Indians"],
        "toss_decision": ["field"],
        "venue": ["Wankhede Stadium"],
        "city": ["Mumbai"]
    }
)

result = model.predict(sample_match)

print("Predicted Winner:", result[0])
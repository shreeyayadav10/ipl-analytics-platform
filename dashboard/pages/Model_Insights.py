
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Model Insights", page_icon="📈", layout="wide")

st.title("📈 Model Insights")

st.metric("Best Model", "Random Forest")
st.metric("Accuracy", "52.75%")
st.metric("Cross Validation", "53.85%")

st.markdown("""
### Feature Engineering Used

- Team 1 Win Rate
- Team 2 Win Rate
- Team 1 Venue Wins
- Head-to-Head Advantage
- Toss Winner
- Toss Decision
- Venue
- City

### Model Comparison

| Model | Accuracy |
|---------|---------|
| Random Forest | 52.75% |
| Gradient Boosting | 51.60% |
| Logistic Regression | 49.80% |

### Cross Validation Scores

- 0.5229
- 0.5642
- 0.5733
- 0.5321
- 0.5000

Average: **0.5385**
""")

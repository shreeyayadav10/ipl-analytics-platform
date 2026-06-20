
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import base64

st.set_page_config(
    page_title="IPL Analytics Platform",
    page_icon="🏏",
    layout="wide"
)

# =========================
# BACKGROUND
# =========================

def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = get_base64("assets/stadium.bg.jpeg")

st.markdown(f"""
<style>
.stApp {{
    background-image:
    linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)),
    url("data:image/jpeg;base64,{bg}");
    background-size: cover;
    background-attachment: fixed;
}}

.big-title {{
    text-align:center;
    font-size:3rem;
    font-weight:bold;
    color:#f59e0b;
}}

.sub-title {{
    text-align:center;
    color:white;
    margin-bottom:20px;
}}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================

model = joblib.load("models/model.pkl")
matches = pd.read_csv("data/processed/featured_matches.csv")

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

TEAM_LOGOS = {
    "Chennai Super Kings":"csk.jpeg",
    "Mumbai Indians":"mi.jpeg",
    "Royal Challengers Bengaluru":"rcb.jpeg",
    "Kolkata Knight Riders":"kkr.jpeg",
    "Delhi Capitals":"dc.jpeg",
    "Punjab Kings":"pbks.jpeg",
    "Rajasthan Royals":"rr.jpeg",
    "Sunrisers Hyderabad":"srh.jpeg",
    "Gujarat Titans":"gt.jpeg",
    "Lucknow Super Giants":"lsg.jpeg"
}

VENUE_CITY_MAP = {
    "Wankhede Stadium":"Mumbai",
    "MA Chidambaram Stadium, Chepauk":"Chennai",
    "Narendra Modi Stadium":"Ahmedabad",
    "M Chinnaswamy Stadium":"Bangalore",
    "Eden Gardens":"Kolkata",
    "Arun Jaitley Stadium":"Delhi",
    "Rajiv Gandhi International Stadium, Uppal":"Hyderabad",
    "Sawai Mansingh Stadium":"Jaipur",
    "Punjab Cricket Association IS Bindra Stadium, Mohali":"Mohali",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium":"Lucknow"
}

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.image("assets/ipl_logo.jpeg", width=180)

    st.title("IPL Analytics")

    st.metric("Dataset Matches", len(matches))
    st.metric("Teams", 10)
    st.metric("Model Accuracy", "52.75%")
    st.metric("CV Score", "53.08%")

    st.markdown("---")

    st.image(
        "assets/ipl_trophy.jpeg",
        use_container_width=True
    )


# =========================
# HEADER
# =========================

st.image("assets/hero_banner.jpeg", use_container_width=True)

st.markdown('<div class="big-title">🏏 IPL Analytics Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Machine Learning Based Match Winner Prediction System</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("Matches", len(matches))
c2.metric("Active Teams", 10)
c3.metric("Accuracy", "52.75%")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🏆 Prediction",
        "📊 Team Analytics",
        "🏟 Venue Analytics",
        "📈 Model Insights",
        "ℹ Project"
    ]
)

# =========================
# PREDICTION TAB
# =========================

with tab1:

    col1, col2 = st.columns(2)

    with col1:
        team1 = st.selectbox("Team 1", ACTIVE_TEAMS)
        venue = st.selectbox("Venue", list(VENUE_CITY_MAP.keys()))

    with col2:
        team2 = st.selectbox("Team 2", ACTIVE_TEAMS)
        toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

    if team1 == team2:
        st.warning("Select different teams.")
        st.stop()

    toss_winner = st.selectbox("Toss Winner", [team1, team2])
    city = VENUE_CITY_MAP[venue]

    left, center, right = st.columns([2,1,2])

    with left:
        st.image(f"assets/{TEAM_LOGOS[team1]}", width=220)

    with center:
        st.markdown("<h1 style='text-align:center;'>VS</h1>", unsafe_allow_html=True)

    with right:
        st.image(f"assets/{TEAM_LOGOS[team2]}", width=220)

    teams_all = pd.unique(pd.concat([matches["team1"], matches["team2"]]))

    rates = {}
    for team in teams_all:
        played = ((matches["team1"] == team).sum() + (matches["team2"] == team).sum())
        wins = (matches["winner"] == team).sum()
        rates[team] = wins / played if played else 0

    team1_win_rate = rates.get(team1, 0)
    team2_win_rate = rates.get(team2, 0)

    s1, s2 = st.columns(2)
    s1.metric(f"{team1} Win Rate", f"{team1_win_rate:.2%}")
    s2.metric(f"{team2} Win Rate", f"{team2_win_rate:.2%}")

    team1_venue_wins = len(
        matches[(matches["winner"] == team1) & (matches["venue"] == venue)]
    )

    h2h = matches[
        (((matches["team1"] == team1) & (matches["team2"] == team2)) |
         ((matches["team1"] == team2) & (matches["team2"] == team1)))
    ]

    team1_h2h = (h2h["winner"] == team1).sum()
    team2_h2h = (h2h["winner"] == team2).sum()

    head_to_head_advantage = team1_h2h - team2_h2h

    h1, h2, h3 = st.columns(3)
    h1.metric(f"{team1} H2H Wins", team1_h2h)
    h2.metric(f"{team2} H2H Wins", team2_h2h)
    h3.metric("Advantage", head_to_head_advantage)

    if st.button("Predict Winner", use_container_width=True):

        sample = pd.DataFrame({
            "team1":[team1],
            "team2":[team2],
            "toss_winner":[toss_winner],
            "toss_decision":[toss_decision],
            "venue":[venue],
            "city":[city],
            "team1_win_rate":[team1_win_rate],
            "team2_win_rate":[team2_win_rate],
            "team1_venue_wins":[team1_venue_wins],
            "head_to_head_advantage":[head_to_head_advantage]
        })

        
        winner = model.predict(sample)[0]
        probs = model.predict_proba(sample)[0]

        c1, c2 = st.columns([1,1])

        with c1:
            st.image(
                f"assets/{TEAM_LOGOS[winner]}",
                width=220
            )

        with c2:
            st.image(
                "assets/ipl_trophy.jpeg",
                width=180
            )

        st.markdown(f"""
        <div style="padding:25px;border-radius:15px;
        background:linear-gradient(90deg,#f59e0b,#dc2626);
        text-align:center;color:white;">
        <h1>🏆 {winner}</h1>
        <h3>Predicted Match Winner</h3>
        </div>
        """, unsafe_allow_html=True)

        # Probability Table
        pdf = pd.DataFrame({
            "Team": model.classes_,
            "Probability": probs * 100
        }).sort_values("Probability", ascending=False)

        # Donut Chart
        fig = px.pie(
            pdf,
            values="Probability",
            names="Team",
            hole=0.65,
            title="Winning Probability Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Probability Breakdown
        st.subheader("Winning Probability Breakdown")
        st.dataframe(pdf, use_container_width=True)

        # Confidence Meter
        confidence = max(probs) * 100
        if confidence >= 70:
            st.success("High Confidence Prediction")
        elif confidence >= 50:
            st.warning("Medium Confidence Prediction")
        else:
            st.error("Low Confidence Prediction")


        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={"text": "Prediction Confidence"},
                gauge={
                    "axis": {"range": [0, 100]}
                }
            )
        )

        st.plotly_chart(gauge, use_container_width=True)

        # Match Insights
        st.subheader("Match Insights")

        feature_df = pd.DataFrame({
            "Feature": [
                "Team 1 Win Rate",
                "Team 2 Win Rate",
                "Team 1 Venue Wins",
                "Head-to-Head Advantage"
            ],
            "Value": [
                f"{team1_win_rate:.2%}",
                f"{team2_win_rate:.2%}",
                team1_venue_wins,
                head_to_head_advantage
            ]
        })

        st.dataframe(
            feature_df,
            use_container_width=True
        )

# =========================
# TEAM ANALYTICS
# =========================

with tab2:

    stats = []

    for team in ACTIVE_TEAMS:
        played = ((matches["team1"] == team).sum() +
                  (matches["team2"] == team).sum())
        wins = (matches["winner"] == team).sum()

        rate = round((wins / played) * 100, 2) if played else 0

        stats.append([team, played, wins, rate])

    team_df = pd.DataFrame(
        stats,
        columns=["Team", "Matches", "Wins", "Win Rate"]
    )

    team_df = team_df.sort_values(
        "Win Rate",
        ascending=False
    ).reset_index(drop=True)

    display_df = team_df.copy()

    display_df["Rank"] = [
        "🥇",
        "🥈",
        "🥉",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10"
    ]

    st.dataframe(
        display_df[
            [
                "Rank",
                "Team",
                "Matches",
                "Wins",
                "Win Rate"
            ]
        ],
        use_container_width=True
    )

    st.success(
        f"🏆 Best Performing Team: {team_df.iloc[0]['Team']} "
        f"({team_df.iloc[0]['Win Rate']}%)"
    )

    fig = px.bar(
        team_df,
        x="Team",
        y="Win Rate",
        color="Win Rate",
        text="Win Rate",
        title="IPL Team Rankings"
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================
# VENUE ANALYTICS
# =========================

with tab3:

    venue_df = matches["venue"].value_counts().head(10).reset_index()
    venue_df.columns = ["Venue", "Matches"]

    st.dataframe(venue_df, use_container_width=True)

    fig = px.bar(
        venue_df,
        x="Venue",
        y="Matches",
        color="Matches",
        text="Matches",
        title="Top IPL Venues"
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
# =========================
# MODEL INSIGHTS TAB
# =========================

with tab4:

    st.header("📈 Model Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Model Accuracy",
            "52.75%"
        )

    with col2:
        st.metric(
            "Cross Validation",
            "53.08%"
        )

    with col3:
        st.metric(
            "Dataset Size",
            "906 Matches"
        )

    st.markdown("---")

    st.subheader(
        "Top Feature Importance"
    )

    st.image(
        "reports/feature_importance.png",
        use_container_width=True
    )

    st.subheader("Feature Importance Summary")

    importance_df = pd.DataFrame(
    {
        "Feature":[
            "Team 1 Win Rate",
            "Head To Head Advantage",
            "Team 2 Win Rate",
            "Venue Wins"
        ],
        "Importance":[
            0.14,
            0.08,
            0.07,
            0.05
        ]
    }
)

    st.dataframe(
        importance_df,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader(
        "Confusion Matrix"
    )

    st.image(
        "reports/confusion_matrix.png",
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Model Comparison")

    comparison_df = pd.DataFrame(
        {
            "Model":[
                "Random Forest",
                "Gradient Boosting",
                "Logistic Regression"
            ],
            "Accuracy":[
                "52.75%",
                "52.29%",
                "51.92%"
            ]
        }
    )

    st.dataframe(
        comparison_df,
        use_container_width=True
    )


    st.markdown("---")

    st.subheader(
        "Key Findings"
    )

    st.success(
    """
    ✔ Team Win Rate is the strongest predictor.

    ✔ Head-to-Head history improves match prediction.

    ✔ Venue performance contributes useful context.

    ✔ Random Forest achieved the best performance.

    ✔ Model trained on 906 IPL matches.

    ✔ Active IPL teams only are used.
    """
)
# =========================
# PROJECT TAB
# =========================

with tab5:

    st.image("assets/pitch.jpeg", use_container_width=True)

    st.metric(
    "Project Completion",
    "Phase 2 Complete"
)

    st.progress(90)

    st.markdown("""
    ## Project Overview

    - IPL Match Winner Prediction
    - Data Cleaning
    - Feature Engineering
    - Random Forest Classifier
    - Streamlit Dashboard
    - Flask Prediction API
    - Model Validation
    - Model Comparison

    ### Tech Stack

    - Python
    - Pandas
    - Scikit-Learn
    - Streamlit
    - Plotly
    - Flask
    """)

    st.markdown("---")

    st.subheader("Project Achievements")

    st.info("""
    🏏 IPL Match Winner Predictor

    📊 Feature Engineering

    🤖 Random Forest Model

    📈 Feature Importance Analysis

    📉 Confusion Matrix Evaluation

    📋 Model Comparison

    🎨 Interactive Dashboard
    """)

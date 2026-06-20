from flask import Flask, jsonify, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load(
    "models/model.pkl"
)


@app.route("/")
def home():
    return jsonify(
        {
            "message": "IPL Prediction API is running"
        }
    )


@app.route("/predict", methods=["POST"])
def predict():

    incoming_data = request.get_json()

    match_details = pd.DataFrame(
        [incoming_data]
    )

    prediction = model.predict(
        match_details
    )

    return jsonify(
        {
            "predicted_winner": prediction[0]
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("models/model.pkl")

@app.route("/")
def home():
    return {
        "message": "IPL Prediction API Running"
    }

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    sample = pd.DataFrame([data])

    prediction = model.predict(sample)[0]

    probabilities = model.predict_proba(sample)[0]

    return jsonify({
        "prediction": prediction,
        "probabilities": {
            team: float(prob)
            for team, prob in zip(
                model.classes_,
                probabilities
            )
        }
    })

if __name__ == "__main__":
    app.run(debug=True)
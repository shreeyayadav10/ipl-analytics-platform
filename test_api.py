import requests

payload = {
    "team1": "Mumbai Indians",
    "team2": "Chennai Super Kings",
    "toss_winner": "Mumbai Indians",
    "toss_decision": "field",
    "venue": "Wankhede Stadium",
    "city": "Mumbai"
}

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json=payload
)

print(response.status_code)
print(response.json())
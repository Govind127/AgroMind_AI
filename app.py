from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import requests
import os

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "crop_model.pkl")
model = joblib.load(MODEL_PATH)

OWM_API_KEY = "4f535c50611950b769c817599a11380e"


def get_weather(city: str):
    if not OWM_API_KEY or OWM_API_KEY == "4f535c50611950b769c817599a11380e":
        return None, "No API key configured."
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={OWM_API_KEY}&units=metric"
        )
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None, f"City not found or API error ({r.status_code})."
        d = r.json()
        weather = {
            "city":        d["name"],
            "country":     d["sys"]["country"],
            "temp":        round(d["main"]["temp"], 1),
            "humidity":    d["main"]["humidity"],
            "description": d["weather"][0]["description"].title(),
            "rainfall":    round(d.get("rain", {}).get("1h", 0.0), 2),
            "wind_speed":  d["wind"]["speed"],
        }
        return weather, None
    except Exception as e:
        return None, str(e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        features = np.array([[
            float(data["N"]),
            float(data["P"]),
            float(data["K"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["ph"]),
            float(data["rainfall"]),
        ]])
        prediction = model.predict(features)[0]
        proba      = model.predict_proba(features)[0]
        confidence = round(float(max(proba)) * 100, 2)

        top3_idx = np.argsort(proba)[::-1][:3]
        top3 = [
            {"crop": model.classes_[i], "prob": round(float(proba[i]) * 100, 2)}
            for i in top3_idx
        ]
        return jsonify({"crop": prediction, "confidence": confidence, "top3": top3})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/weather", methods=["POST"])
def weather():
    data = request.get_json()
    city = data.get("city", "").strip()
    if not city:
        return jsonify({"error": "Please enter a city name."}), 400
    result, err = get_weather(city)
    if err:
        return jsonify({"error": err}), 400
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
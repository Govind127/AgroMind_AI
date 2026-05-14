# AgroMind_AI
A perfect destination for farmer's crop recommendation. A one stop shop for all farmers with Weather API integration!!!

# AgroMind AI — Crop Intelligence Dashboard

A machine learning web application that recommends the most suitable crop to grow based on soil nutrient levels and real-time weather conditions. Built with Flask, scikit-learn Random Forest, and the OpenWeatherMap API.

---

## What It Does

- Takes soil nutrient inputs (Nitrogen, Phosphorus, Potassium) and climate parameters (temperature, humidity, pH, rainfall)
- Fetches **live weather data** from OpenWeatherMap for any city and auto-fills the climate fields
- Runs a trained **Random Forest Classifier** to predict the best crop
- Displays the top 3 crop recommendations with confidence scores and animated probability bars

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML Model | scikit-learn Random Forest Classifier |
| Dataset | UCI Crop Recommendation Dataset (2200 samples, 22 crop classes) |
| Weather | OpenWeatherMap Current Weather API |
| Frontend | HTML, CSS, Vanilla JS (no frameworks) |

---

## Project Structure

```
agromind/
├── app.py                  # Flask backend, prediction and weather routes
├── crop_model.pkl          # Trained Random Forest model
├── Crop_recommendation.csv # Training dataset
├── requirements.txt        # Python dependencies
└── templates/
    └── index.html          # Dashboard UI
```

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/agromind-ai.git
cd agromind-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get a free Weather API key

- Sign up at [openweathermap.org](https://openweathermap.org) (free tier)
- Go to **API Keys** in your dashboard and copy your key
- Open `app.py` and replace line 13:

```python
OWM_API_KEY = "your_api_key_here"
```

> Note: New API keys take 10–60 minutes to activate after signup.

### 4. Run the app

```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000`

---

## Model Details

- **Algorithm:** Random Forest Classifier
- **Dataset:** UCI Crop Recommendation Dataset
- **Features:** N, P, K, temperature, humidity, pH, rainfall
- **Target Classes:** 22 crop types including rice, maize, wheat, mango, banana, coffee, cotton, and more
- **Training split:** 80/20 train-test

---

## How the Weather Integration Works

The live weather feature uses OpenWeatherMap's Current Weather API to fetch real environmental data for any city. Once fetched, clicking **Use Live Weather** auto-fills the temperature, humidity, and rainfall sliders — making the crop prediction reflect actual field conditions rather than manual estimates.

---

## Requirements

```
flask
joblib
numpy
scikit-learn
requests
```

---

## Dataset

The dataset is sourced from the [UCI Machine Learning Repository](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset). It contains 2200 records across 22 crop classes with 7 numeric features each.

---

## Author

**Govind** — Data Science Student, Pune  
Built as part of a major project integrating ML with real-world agriculture use cases.

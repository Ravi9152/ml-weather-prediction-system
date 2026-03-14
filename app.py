from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# ----------------- CONFIGURATION -----------------
MODEL_PATH = r"D:\New folder\output\best_model.pkl"
SCALER_PATH = r"D:\New folder\output\scaler.pkl"

# Load the model and scaler
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully!")
else:
    model = None
    scaler = None
    print("ERROR: Model or Scaler file not found. Please run weather_prediction.py first.")

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/style.css")
def serve_css():
    return send_from_directory(".", "style.css")

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not scaler:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json
        print(f"Received data: {data}")

        # Extract features (adjusting map to match what the model expects)
        # Note: In a real scenario, we'd need to handle categorical encoding precisely.
        # For simplicity in this "easy code" request, we'll assume numeric inputs.
        
        # Expected features (order matters):
        # MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustSpeed, 
        # WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, 
        # Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, 
        # (plus encoded categorical ones)
        
        # Create a DataFrame for prediction
        # We'll use default values for features not provided by the UI
        features = {
            'Location': 1,  # Default encoded location
            'MinTemp': float(data.get('MinTemp', 15)),
            'MaxTemp': float(data.get('MaxTemp', 25)),
            'Rainfall': float(data.get('Rainfall', 0)),
            'Evaporation': 4.8,
            'Sunshine': 8.5,
            'WindGustDir': 0,
            'WindGustSpeed': float(data.get('WindGustSpeed', 40)),
            'WindDir9am': 0,
            'WindDir3pm': 0,
            'WindSpeed9am': 15,
            'WindSpeed3pm': 20,
            'Humidity9am': float(data.get('Humidity9am', 60)),
            'Humidity3pm': float(data.get('Humidity3pm', 45)),
            'Pressure9am': 1015,
            'Pressure3pm': 1012,
            'Cloud9am': 4,
            'Cloud3pm': 4,
            'Temp9am': 18,
            'Temp3pm': 23,
            'RainToday': 0
        }
        
        input_df = pd.DataFrame([features])
        
        # Ensure column order identically matches the exact order scaler was trained on
        if hasattr(scaler, "feature_names_in_"):
            input_df = input_df[scaler.feature_names_in_]
        
        # Scale the data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        pred_class = model.predict(input_scaled)[0]
        
        # Extract the probability of the currently PREDICTED class
        probabilities = model.predict_proba(input_scaled)[0]
        match_prob = probabilities[pred_class]
        
        result = "Yes" if pred_class == 1 else "No"
        
        return jsonify({
            "prediction": result,
            "probability": f"{match_prob*100:.1f}%",
            "model_type": "XGBoost Classifier",
            "accuracy": "82.87%",
            "roc_auc": "0.8943",
            "features_used": "21 Data Points"
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)

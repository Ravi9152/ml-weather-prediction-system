# ML Weather Prediction System

A Machine Learning based Weather Prediction Web Application built using **Python, Flask, and XGBoost**.
This system predicts the **probability of rain** based on meteorological parameters such as temperature, humidity, rainfall, and wind speed.

---

## 🚀 Features

* Machine Learning rain prediction using **XGBoost Classifier**
* Interactive **AI dashboard UI**
* Rain probability visualization
* Flask based backend API
* Real-time prediction using input parameters

---

## 🧠 Machine Learning Model

Algorithm Used:

XGBoost Classifier

Model Metrics:

Accuracy: ~82%
ROC AUC Score: ~0.89

Dataset:

Meteorological weather dataset used to train the model.

---

## 🖥️ Tech Stack

* Python
* Flask
* Scikit-learn
* XGBoost
* HTML
* CSS
* JavaScript

---

## 📂 Project Structure

```
ml-weather-prediction-system
│
app.py
weather_prediction.py
requirements.txt
README.md
│
templates
│   index.html
│
static
    style.css
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/Ravi9152/ml-weather-prediction-system.git
```

Go to the project directory:

```
cd ml-weather-prediction-system
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 📊 Input Parameters

The model uses the following inputs:

* Minimum Temperature
* Maximum Temperature
* Rainfall
* Wind Gust Speed
* Humidity at 9am
* Humidity at 3pm

Based on these parameters, the model predicts the **probability of rain**.


## Project Dashbord
<p align="center">
<img src="ml weather.png" width="800">
</p>

---

## 👨‍💻 Author

Ravi
B.Tech Student | Machine Learning Enthusiast

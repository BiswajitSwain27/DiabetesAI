from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import xgboost as xgb
import pandas as pd 

app = Flask(__name__)
CORS(app)

# ── Load XGBoost model using native JSON format (no version warning) ──
model = xgb.XGBClassifier()
model.load_model('xgboost_model.json')
print("✅ XGBoost model loaded successfully!")

# ── Load scaler ──
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("✅ Scaler loaded successfully!")

# ── Serve the dashboard ──
@app.route('/')
def home():
    return render_template('index.html')

# ── Prediction API endpoint ──
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = [
            float(data['pregnancies']),
            float(data['glucose']),
            float(data['bloodpressure']),
            float(data['skinthickness']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['diabetespedigree']),
            float(data['age'])
        ]

        features_array  = pd.DataFrame([features], columns=[
            'Pregnancies', 'Glucose', 'BloodPressure',
            'SkinThickness', 'Insulin', 'BMI',
            'DiabetesPedigreeFunction', 'Age'
        ])
        features_scaled = scaler.transform(features_array)

        prediction  = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        non_diabetic_prob = round(float(probability[0]) * 100, 1)
        diabetic_prob     = round(float(probability[1]) * 100, 1)

        return jsonify({
            'success':                 True,
            'prediction':              int(prediction),
            'label':                   'Diabetic' if prediction == 1 else 'Non-Diabetic',
            'diabetic_probability':    diabetic_prob,
            'non_diabetic_probability': non_diabetic_prob,
            'confidence':              diabetic_prob if prediction == 1 else non_diabetic_prob
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    print("🚀 Starting Diabetes Prediction API...")
    print("📊 Dashboard: http://localhost:5000")
    app.run(debug=True, port=5000)
# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model and scaler at startup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir # Assuming app.py is in the root directory

model_path = os.path.join(project_root, 'model', 'model.joblib')
scaler_path = os.path.join(project_root, 'data', 'processed', 'scaler.joblib')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Warning: Error loading model/scaler: {e}. Please ensure preprocess.py and train.py have been run.")
    model = None
    scaler = None

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({'error': 'Model or scaler not loaded properly.'}), 500
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
            
        # Convert JSON data to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
            
        # Expected features in the exact order the model expects
        expected_features = [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ]
        
        # Ensure all expected features are present
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'}), 400
            
        # Reorder columns to match expected order
        X = df[expected_features]
        
        # Scale features
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
        
        # Predict
        predictions = model.predict(X_scaled)
        
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

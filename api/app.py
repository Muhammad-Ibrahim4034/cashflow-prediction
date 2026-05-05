from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load models
MODEL_PATH = "models/best_cashflow_model.pkl"
PIPELINE_PATH = "models/feature_pipeline.pkl"

model = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    
    # Process through pipeline
    processed_data = pipeline.transform(df)
    
    # Predict
    prediction = model.predict(processed_data)
    
    return jsonify({
        "predicted_net_cashflow": float(prediction[0])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

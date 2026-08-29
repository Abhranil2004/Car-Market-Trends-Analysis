import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_car_model.pkl')
SUMMARY_PATH = os.path.join(BASE_DIR, 'model', 'summary_data.json')
CSV_PATH = os.path.join(os.path.dirname(BASE_DIR), '1776311302-P3-Car Market Trends Analysis with Car Dekho Data.csv')

# Load trained pipeline
pipeline = None
if os.path.exists(MODEL_PATH):
    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        print("Error loading model:", e)

# Load summary
summary_data = {}
if os.path.exists(SUMMARY_PATH):
    with open(SUMMARY_PATH, 'r') as f:
        summary_data = json.load(f)

# Load raw sample for table
df_cars = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html', summary=summary_data)

@app.route('/api/data')
def get_data():
    return jsonify(summary_data)

@app.route('/api/cars')
def get_cars():
    fuel = request.args.get('fuel', 'All')
    trans = request.args.get('transmission', 'All')
    seller = request.args.get('seller', 'All')
    
    filtered = df_cars.copy()
    if fuel != 'All':
        filtered = filtered[filtered['Fuel_Type'] == fuel]
    if trans != 'All':
        filtered = filtered[filtered['Transmission'] == trans]
    if seller != 'All':
        filtered = filtered[filtered['Seller_Type'] == seller]
        
    records = filtered.head(50).to_dict(orient='records')
    return jsonify({
        'total': len(filtered),
        'cars': records
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        present_price = float(data.get('present_price', 5.0))
        kms_driven = int(data.get('kms_driven', 30000))
        fuel_type = str(data.get('fuel_type', 'Petrol'))
        seller_type = str(data.get('seller_type', 'Dealer'))
        transmission = str(data.get('transmission', 'Manual'))
        owner = int(data.get('owner', 0))
        year = int(data.get('year', 2018))
        
        car_age = 2026 - year
        
        input_df = pd.DataFrame([{
            'Present_Price': present_price,
            'Kms_Driven': kms_driven,
            'Fuel_Type': fuel_type,
            'Seller_Type': seller_type,
            'Transmission': transmission,
            'Owner': owner,
            'Car_Age': car_age
        }])
        
        pred_price = pipeline.predict(input_df)[0]
        pred_price = max(0.2, round(float(pred_price), 2))
        
        # Calculate depreciation metrics
        deprec_val = round(present_price - pred_price, 2)
        deprec_pct = round(((present_price - pred_price) / present_price) * 100, 1)
        
        return jsonify({
            'success': True,
            'predicted_selling_price': pred_price,
            'depreciation_value': deprec_val,
            'depreciation_percent': deprec_pct,
            'currency': 'Lakhs INR',
            'car_age': car_age
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)

import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

app = Flask(__name__)

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_car_model.pkl')
SUMMARY_PATH = os.path.join(BASE_DIR, 'model', 'summary_data.json')
CSV_PATH = os.path.join(os.path.dirname(BASE_DIR), '1776311302-P3-Car Market Trends Analysis with Car Dekho Data.csv')

# Load raw dataset
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(BASE_DIR, '1776311302-P3-Car Market Trends Analysis with Car Dekho Data.csv')

df_cars = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame()

# Load summary
summary_data = {}
if os.path.exists(SUMMARY_PATH):
    try:
        with open(SUMMARY_PATH, 'r') as f:
            summary_data = json.load(f)
    except Exception as e:
        print("Summary load error:", e)

# Preprocessor & Pipeline Trainer Helper
def train_fresh_pipeline(df):
    current_year = 2026
    df_train = df.copy()
    df_train['Car_Age'] = current_year - df_train['Year']
    
    X = df_train[['Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner', 'Car_Age']]
    y = df_train['Selling_Price']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Present_Price', 'Kms_Driven', 'Owner', 'Car_Age']),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['Fuel_Type', 'Seller_Type', 'Transmission'])
        ]
    )
    
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    pipe.fit(X, y)
    return pipe

# Load or dynamically build pipeline
pipeline = None
if os.path.exists(MODEL_PATH):
    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        print("Pickle version warning, training fresh in-memory model:", e)

if pipeline is None and not df_cars.empty:
    try:
        pipeline = train_fresh_pipeline(df_cars)
        print("In-memory ML model trained successfully.")
    except Exception as e:
        print("Error training model:", e)

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
    
    if df_cars.empty:
        return jsonify({'total': 0, 'cars': []})
        
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
        
        if pipeline is not None:
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
        else:
            # Fallback valuation equation
            deprec_rate = 0.12 * car_age + (kms_driven / 100000) * 0.05
            if fuel_type == 'Diesel': deprec_rate *= 0.88
            if transmission == 'Automatic': deprec_rate *= 0.92
            if seller_type == 'Individual': deprec_rate *= 1.15
            deprec_rate = min(0.85, max(0.1, deprec_rate))
            pred_price = present_price * (1 - deprec_rate)
            
        pred_price = max(0.2, round(float(pred_price), 2))
        
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

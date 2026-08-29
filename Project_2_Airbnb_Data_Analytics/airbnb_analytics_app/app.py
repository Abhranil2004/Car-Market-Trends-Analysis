import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SUMMARY_PATH = os.path.join(DATA_DIR, 'airbnb_summary.json')
MODEL_PATH = os.path.join(DATA_DIR, 'airbnb_price_model.pkl')
SAMPLE_CSV_PATH = os.path.join(DATA_DIR, 'airbnb_sample.csv')

# Load summary
summary_data = {}
if os.path.exists(SUMMARY_PATH):
    try:
        with open(SUMMARY_PATH, 'r') as f:
            summary_data = json.load(f)
    except Exception as e:
        print("Summary load error:", e)

# Load model
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print("Model load error:", e)

# Load sample
df_sample = pd.read_csv(SAMPLE_CSV_PATH) if os.path.exists(SAMPLE_CSV_PATH) else pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html', summary=summary_data)

@app.route('/api/summary')
def get_summary():
    return jsonify(summary_data)

@app.route('/api/listings')
def get_listings():
    borough = request.args.get('borough', 'All')
    room_type = request.args.get('room_type', 'All')
    
    if df_sample.empty:
        return jsonify({'total': 0, 'listings': []})
        
    filtered = df_sample.copy()
    if borough != 'All':
        filtered = filtered[filtered['neighbourhood_group'] == borough]
    if room_type != 'All':
        filtered = filtered[filtered['room_type'] == room_type]
        
    records = filtered.head(50).to_dict(orient='records')
    return jsonify({
        'total': len(filtered),
        'listings': records
    })

@app.route('/api/estimate_price', methods=['POST'])
def estimate_price():
    try:
        data = request.get_json(force=True)
        borough = str(data.get('borough', 'Manhattan'))
        room_type = str(data.get('room_type', 'Entire home/apt'))
        min_nights = int(data.get('minimum_nights', 3))
        availability = int(data.get('availability_365', 180))
        rating = float(data.get('review_rate_number', 4.5))
        
        # Base estimate fallback if model not loaded
        if model is not None:
            input_df = pd.DataFrame([{
                'neighbourhood_group': borough,
                'room_type': room_type,
                'minimum_nights': min_nights,
                'availability_365': availability,
                'review_rate_number': rating
            }])
            est_price = model.predict(input_df)[0]
        else:
            base_rates = {'Manhattan': 220, 'Brooklyn': 150, 'Queens': 110, 'Bronx': 90, 'Staten Island': 95}
            room_mult = {'Entire home/apt': 1.6, 'Private room': 0.85, 'Shared room': 0.45, 'Hotel room': 2.1}
            est_price = base_rates.get(borough, 130) * room_mult.get(room_type, 1.0)
            
        est_price = max(35, round(float(est_price), 2))
        est_service_fee = round(est_price * 0.17, 2)
        
        return jsonify({
            'success': True,
            'estimated_nightly_price': est_price,
            'estimated_service_fee': est_service_fee,
            'currency': 'USD',
            'recommended_price_range': f"${max(25, int(est_price * 0.85))} - ${int(est_price * 1.15)}"
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=True)

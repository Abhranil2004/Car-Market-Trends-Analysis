import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

def process_and_export():
    print("Reading Airbnb_Open_Data.xlsx...")
    df = pd.read_excel('Airbnb_Open_Data.xlsx')
    print(f"Loaded raw dataset with shape: {df.shape}")
    
    # 1. Deduplication
    df = df.drop_duplicates().copy()
    
    # 2. Fix Column Names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    
    # 3. Clean Price and Service Fee columns if they are strings with '$' or ','
    for col in ['price', 'service_fee']:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows without price or geographic coordinates
    df = df.dropna(subset=['price', 'lat', 'long']).copy()
    
    # 4. Standardize Neighbourhood Groups
    if 'neighbourhood_group' in df.columns:
        df['neighbourhood_group'] = df['neighbourhood_group'].astype(str).str.strip()
        df['neighbourhood_group'] = df['neighbourhood_group'].replace({
            'manhatan': 'Manhattan',
            'brookln': 'Brooklyn'
        })
        # Keep valid boroughs
        valid_boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
        df = df[df['neighbourhood_group'].isin(valid_boroughs)].copy()
        
    # 5. Clean & Impute Categorical & Numeric Fields
    df['name'] = df['name'].fillna('Unknown Listing')
    df['host_name'] = df['host_name'].fillna('Unknown Host')
    df['host_identity_verified'] = df['host_identity_verified'].fillna('unconfirmed')
    df['cancellation_policy'] = df['cancellation_policy'].fillna('strict')
    df['room_type'] = df['room_type'].fillna('Entire home/apt')
    
    df['number_of_reviews'] = df['number_of_reviews'].fillna(0).astype(int)
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0.0)
    df['review_rate_number'] = df['review_rate_number'].fillna(df['review_rate_number'].median())
    
    # Sanitize minimum nights and availability
    df.loc[~df['minimum_nights'].between(1, 365), 'minimum_nights'] = np.nan
    df['minimum_nights'] = df['minimum_nights'].fillna(df['minimum_nights'].median()).astype(int)
    
    df.loc[~df['availability_365'].between(0, 365), 'availability_365'] = np.nan
    df['availability_365'] = df['availability_365'].fillna(df['availability_365'].median()).astype(int)
    
    df['construction_year'] = df['construction_year'].fillna(df['construction_year'].median()).astype(int)
    if 'service_fee' in df.columns:
        df['service_fee'] = df['service_fee'].fillna(df['price'] * 0.2)
        
    print(f"Cleaned dataset shape: {df.shape}")
    
    # Save a lightweight cleaned CSV sample for fast web table preview
    sample_df = df[['name', 'host_name', 'neighbourhood_group', 'neighbourhood', 'room_type', 
                    'price', 'service_fee', 'minimum_nights', 'number_of_reviews', 'review_rate_number', 'availability_365']].sample(n=min(5000, len(df)), random_state=42)
    os.makedirs('airbnb_analytics_app/data', exist_ok=True)
    sample_df.to_csv('airbnb_analytics_app/data/airbnb_sample.csv', index=False)
    
    # Compute Analytics Summary
    borough_stats = df.groupby('neighbourhood_group').agg(
        listing_count=('price', 'count'),
        avg_price=('price', 'mean'),
        median_price=('price', 'median'),
        avg_availability=('availability_365', 'mean'),
        avg_reviews=('number_of_reviews', 'mean')
    ).round(2).to_dict(orient='index')
    
    room_stats = df.groupby('room_type').agg(
        listing_count=('price', 'count'),
        avg_price=('price', 'mean'),
        avg_availability=('availability_365', 'mean')
    ).round(2).to_dict(orient='index')
    
    top_neighbourhoods = df['neighbourhood'].value_counts().head(10).to_dict()
    
    price_ranges = {
        'Budget ($0-$250)': int((df['price'] <= 250).sum()),
        'Moderate ($251-$600)': int(((df['price'] > 250) & (df['price'] <= 600)).sum()),
        'Premium ($601-$900)': int(((df['price'] > 600) & (df['price'] <= 900)).sum()),
        'Luxury ($901+)': int((df['price'] > 900).sum())
    }
    
    summary = {
        'total_listings': int(len(df)),
        'avg_price': round(float(df['price'].mean()), 2),
        'median_price': round(float(df['price'].median()), 2),
        'avg_reviews': round(float(df['number_of_reviews'].mean()), 1),
        'avg_availability': round(float(df['availability_365'].mean()), 1),
        'borough_stats': borough_stats,
        'room_stats': room_stats,
        'top_neighbourhoods': top_neighbourhoods,
        'price_ranges': price_ranges,
        'borough_list': list(borough_stats.keys()),
        'room_types': list(room_stats.keys())
    }
    
    with open('airbnb_analytics_app/data/airbnb_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    print("Airbnb summary exported.")
    
    # Train Listing Price Benchmark Model
    feature_cols = ['neighbourhood_group', 'room_type', 'minimum_nights', 'availability_365', 'review_rate_number']
    X = df[feature_cols]
    y = df['price']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['neighbourhood_group', 'room_type'])
        ],
        remainder='passthrough'
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    print(f"Price Estimator Model trained. MAE: ${mean_absolute_error(y_test, preds):.2f}")
    
    joblib.dump(model, 'airbnb_analytics_app/data/airbnb_price_model.pkl')
    print("Model saved to airbnb_analytics_app/data/airbnb_price_model.pkl")

if __name__ == '__main__':
    process_and_export()

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def train_and_evaluate():
    csv_path = '1776311302-P3-Car Market Trends Analysis with Car Dekho Data.csv'
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Feature Engineering
    current_year = 2026
    df['Car_Age'] = current_year - df['Year']
    df['Price_Ratio'] = (df['Selling_Price'] / df['Present_Price']).round(4)
    df['Log_Kms_Driven'] = np.log1p(df['Kms_Driven'])
    
    # Target and Features
    X = df[['Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner', 'Car_Age']]
    y = df['Selling_Price']
    
    categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission']
    numerical_features = ['Present_Price', 'Kms_Driven', 'Owner', 'Car_Age']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Models to train
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_r2 = -float('inf')
    best_pipeline = None
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
        
        results[name] = {
            'R2': round(float(r2), 4),
            'RMSE': round(float(rmse), 4),
            'MAE': round(float(mae), 4),
            'CV_R2_Mean': round(float(cv_scores.mean()), 4),
            'CV_R2_Std': round(float(cv_scores.std()), 4)
        }
        print(f"[{name}] Test R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | CV R2: {cv_scores.mean():.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline
            
    print(f"\nBest Model: {best_model_name} with Test R2: {best_r2:.4f}")
    
    # Save model and artifacts
    os.makedirs('car_market_app/model', exist_ok=True)
    joblib.dump(best_pipeline, 'car_market_app/model/best_car_model.pkl')
    
    # Feature Importance for Tree Model
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
    rf_pipeline.fit(X, y)
    
    cat_encoder = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    all_feature_names = numerical_features + cat_names
    importances = rf_pipeline.named_steps['regressor'].feature_importances_
    
    feat_imp = sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)
    
    # Prepare summary data for the web app
    app_data = {
        'total_cars': int(len(df)),
        'avg_selling_price': round(float(df['Selling_Price'].mean()), 2),
        'avg_present_price': round(float(df['Present_Price'].mean()), 2),
        'avg_kms': round(float(df['Kms_Driven'].mean()), 0),
        'fuel_types': df['Fuel_Type'].value_counts().to_dict(),
        'seller_types': df['Seller_Type'].value_counts().to_dict(),
        'transmissions': df['Transmission'].value_counts().to_dict(),
        'model_results': results,
        'best_model': best_model_name,
        'feature_importance': [{'feature': f, 'importance': round(float(imp), 4)} for f, imp in feat_imp],
        'price_by_fuel': df.groupby('Fuel_Type')['Selling_Price'].mean().round(2).to_dict(),
        'price_by_transmission': df.groupby('Transmission')['Selling_Price'].mean().round(2).to_dict(),
        'price_by_seller': df.groupby('Seller_Type')['Selling_Price'].mean().round(2).to_dict(),
        'avg_price_by_year': df.groupby('Year')['Selling_Price'].mean().round(2).to_dict()
    }
    
    with open('car_market_app/model/summary_data.json', 'w') as f:
        json.dump(app_data, f, indent=2)
        
    print("Model and summary artifacts saved successfully.")

if __name__ == '__main__':
    train_and_evaluate()

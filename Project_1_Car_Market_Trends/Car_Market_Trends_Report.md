# 🚗 Car Market Trends Analysis & Used Vehicle Valuation Engine
## **Internship Project Technical Report & Executive Summary**

---

### **Project Metadata**
- **Project Title:** Car Market Trends Analysis with Car Dekho Data
- **Dataset:** `1776311302-P3-Car Market Trends Analysis with Car Dekho Data.csv` (301 records, 9 features)
- **Role:** Data Analytics & Machine Learning Intern
- **Key Technologies:** Python 3, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Flask, HTML5/CSS3, Chart.js

---

## 1. Executive Summary
This project investigates the automotive secondary market using historical sales data from **Car Dekho**. The primary objective is to identify key macroeconomic and vehicle-specific drivers of used car depreciation and resale prices, construct predictive valuation models, and deliver an interactive web dashboard for real-time fair market valuation.

### Key Milestones Achieved:
1. **End-to-End Data Pipeline:** Rigorous cleaning, integrity checks, and validation of vehicle specifications.
2. **Feature Engineering:** Calculated `Car_Age`, `Depreciation_Value`, `Depreciation_Percent`, and `Price_Retention_Ratio`.
3. **Exploratory Data Analysis (EDA):** Discovered nonlinear depreciation curves, transmission premiums, and fuel type dynamics.
4. **Machine Learning Benchmarking:** Evaluated 4 regression algorithms. **Gradient Boosting Regressor** achieved superior performance with **$R^2 = 0.9694$** and **MAE = 0.559 Lakhs INR**.
5. **Interactive Web Dashboard:** Deployed a glassmorphic web interface with Chart.js analytics and dynamic price calculator.

---

## 2. Dataset Overview & Features

| Feature Name | Description | Data Type | Notes |
| :--- | :--- | :--- | :--- |
| `Car_Name` | Make & Model of vehicle | Categorical | 98 unique models |
| `Year` | Manufacturing Year | Numeric (Integer) | Ranging from 2003 to 2018 |
| `Selling_Price` | Resale Price (Target) | Numeric (Float) | Units: Lakhs INR (₹100,000) |
| `Present_Price` | Original Ex-Showroom Price | Numeric (Float) | Units: Lakhs INR |
| `Kms_Driven` | Total distance traveled | Numeric (Integer) | 500 to 500,000 km |
| `Fuel_Type` | Fuel powertrain | Categorical | Petrol, Diesel, CNG |
| `Seller_Type` | Sales channel | Categorical | Dealer (Certified), Individual |
| `Transmission` | Gearbox configuration | Categorical | Manual, Automatic |
| `Owner` | Previous ownership count | Numeric (Integer) | 0 (First owner) to 3 |

---

## 3. Exploratory Data Analysis (EDA) Insights

### 3.1 Price Depreciation vs. Vehicle Age
- Used car valuations follow a steep initial depreciation curve, losing **30–40% of their original showroom value within the first 3–4 years**.
- Beyond 7 years, depreciation flattens as the vehicle approaches its baseline utility value.

### 3.2 Fuel Powertrain Performance
- **Diesel Vehicles:** Command higher average resale prices (avg ₹10.27 Lakhs) compared to Petrol (avg ₹3.26 Lakhs), primarily driven by commercial demand and fuel efficiency.
- **CNG Vehicles:** Exhibit lower resale value but stable budget demand.

### 3.3 Transmission & Seller Channel Premiums
- **Automatic Transmissions:** Demand an average price premium of **₹9.42 Lakhs** vs. **₹3.93 Lakhs** for manual gearboxes.
- **Dealer Listings:** Sell for an average of **₹6.72 Lakhs** compared to **₹1.47 Lakhs** for individual owner listings due to certified multi-point inspections and dealer warranties.

---

## 4. Machine Learning Modeling & Evaluation

The dataset was split into **80% Training** and **20% Testing** sets with standard scaling on continuous variables and one-hot encoding on categorical attributes.

### Model Benchmarking Results:

| Model | Test $R^2$ Score | Root Mean Squared Error (RMSE) | Mean Absolute Error (MAE) | Rank |
| :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting Regressor** | **0.9694 (96.9%)** | **0.8393 Lakhs** | **0.5590 Lakhs** | 🥇 **Best Model** |
| **Random Forest Regressor** | **0.9621 (96.2%)** | **0.9342 Lakhs** | **0.6218 Lakhs** | 🥈 Runner Up |
| **Ridge Regression (L2)** | **0.8492 (84.9%)** | **1.8637 Lakhs** | **1.2138 Lakhs** | 🥉 Linear Regularized |
| **Linear Regression** | **0.8490 (84.9%)** | **1.8652 Lakhs** | **1.2162 Lakhs** | 4th Baseline |

### Feature Importance (Random Forest Gini Impurity):
1. **`Present_Price` (Showroom Price):** **86.4%** relative importance.
2. **`Car_Age` (Vehicle Age):** **8.2%** relative importance.
3. **`Kms_Driven` (Mileage):** **3.1%** relative importance.
4. **`Seller_Type_Individual`:** **1.1%** relative importance.
5. **`Transmission_Manual`:** **0.7%** relative importance.
6. **`Fuel_Type_Diesel`:** **0.5%** relative importance.

---

## 5. Deployment & Web Application
The trained pipeline was packaged into `car_market_app/app.py`:
- **Port:** `5001`
- **Features:** Real-time valuation calculator, Chart.js visual trends, and dataset filter explorer.

To launch:
```bash
python car_market_app/app.py
```
Visit `http://localhost:5001` in your browser.

---

## 6. Recommendations & Business Value
1. **For Dealerships:** Focus inventory acquisition on 3–5 year-old diesel and automatic SUVs, which offer the highest margin-to-holding-period ratio.
2. **For Consumers:** Best value retention is found in certified 4-year-old manual petrol sedans.

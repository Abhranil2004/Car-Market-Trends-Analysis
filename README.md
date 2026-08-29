# 🎓 Data Analytics & Machine Learning Internship Projects

This repository contains two complete, production-grade Data Analytics & Machine Learning projects built as part of the internship program:

---

## 📁 Repository Structure

```
Internship Projects/
│
├── 🚗 Project 1: Car Market Trends Analysis (Car Dekho)
│   ├── 1776311302-P3-Car Market Trends Analysis with Car Dekho Data.csv  # Raw Dataset
│   ├── Car_Market_Trends_Analysis.ipynb                                  # Complete End-to-End Notebook
│   ├── train_car_model.py                                               # ML Training & Benchmark Script
│   ├── Car_Market_Trends_Report.md                                      # Full Formal Technical Report
│   └── car_market_app/                                                  # Interactive Web Application
│       ├── app.py                                                       # Flask Server & Valuation API (Port 5001)
│       ├── templates/index.html                                         # Glassmorphic UI & Visual Charts
│       ├── static/style.css                                             # Styling & Responsive Layout
│       └── model/                                                       # Trained Model Weights & Encoders
│
├── 🏙️ Project 2: NYC Airbnb Data Analytics & Market Intelligence
│   ├── Airbnb_Open_Data.xlsx                                            # Raw NYC Dataset (100k+ rows)
│   ├── Airbnb_Data_Analytics.ipynb                                      # Upgraded & Complete Notebook
│   ├── process_airbnb_data.py                                           # Data Cleaning & Processor Script
│   ├── Airbnb_Market_Analytics_Report.md                                # Full Formal Technical Report
│   └── airbnb_analytics_app/                                            # Interactive Web Application
│       ├── app.py                                                       # Flask Server & Price Estimator (Port 5002)
│       ├── templates/index.html                                         # Interactive Dashboard & Charts
│       ├── static/style.css                                             # Styling & Visual Design
│       └── data/                                                        # Cleaned Sample & ML Model
│
└── README.md                                                            # Repository Guide & Documentation
```

---

## 🚀 How to Run the Projects

### 1. Car Market Trends Analysis Web App
```bash
# In the repository root
python car_market_app/app.py
```
Open **[http://localhost:5001](http://localhost:5001)** in your browser to access the interactive Car Dekho Valuation Engine and Market Trends Dashboard.

---

### 2. NYC Airbnb Analytics Web App
```bash
# In the repository root
python airbnb_analytics_app/app.py
```
Open **[http://localhost:5002](http://localhost:5002)** in your browser to access the NYC Airbnb Market Intelligence Dashboard and Nightly Rate Estimator.

---

### 3. Jupyter Notebooks
Both notebooks can be opened and executed sequentially in Jupyter Lab, Jupyter Notebook, VS Code, or Google Colab:
- **`Car_Market_Trends_Analysis.ipynb`**
- **`Airbnb_Data_Analytics.ipynb`**

---

## 📊 Summary of Results

| Project | Dataset Size | Primary Model | Test Performance | Deliverables |
| :--- | :--- | :--- | :--- | :--- |
| **Car Market Trends** | 301 cars, 9 features | Gradient Boosting Regressor | **$R^2 = 0.9694$**, MAE = 0.559 Lakhs | Notebook, Web App, Technical Report |
| **NYC Airbnb Analytics** | 102,599 listings, 26 features | Random Forest Regressor | Spatial & Pricing Intelligence | Notebook, Web App, Technical Report |

---

## 👨‍💻 Author & Submission
- **Internship Project Submission**
- Complete source code, models, visualizations, reports, and web interfaces included.

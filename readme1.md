# 💼 Salary Prediction using ML Pipeline & MLflow

This project aims to build a robust machine learning pipeline that predicts adjusted salary (`adjusted_total_usd`) from structured job data using modern tools such as MLflow, Evidently AI, PostgreSQL, and SHAP for interpretability.

---

## 📦 Project Structure

Final_Inslit_HR/
├── app.py                          # Flask main application
├── predict.py                      # Prediction logic using MLflow model
├── requirements.txt                # Dependencies
├── config.py                       # Configuration (DB, paths)
├── .env                            # Environment variables
│
├── data/
│   └── load_data.py                # CSV → DataFrame
│
├── db/
│   └── csv_to_sql.py               # Load CSV to PostgreSQL
│
├── features/
│   ├── preprocess.py               # Data preprocessing
│   ├── feature_engineering.py      # Feature creation like `is_remote`
│   └── feature_selection.py        # RFE, RF-based feature selection
│
├── training/
│   └── train_model.py              # Model training & MLflow logging
│
├── analysis/
│   └── eda.py                      # Exploratory Data Analysis
│
├── mlruns/                         # MLflow experiment logs
│
├── sitecustomize.py                # Custom patch for YAML if needed
└── templates/
    └── index.html                  # Frontend (file upload form)



---

## 📌 Setup Instructions

1. **Install Environment (with conda)**
conda create -n inslit-hr -c conda-forge python=3.11 numpy=1.22.4 scipy=1.8.1 scikit-learn=1.7.0 xgboost numba shap pandas matplotlib seaborn python-dotenv psycopg2 sqlalchemy mlflow evidently -y
conda activate inslit-hr

Install Flask and Other Requirements
pip install flask
pip install -r requirements.txt


Fix MLflow YAML Bug (if needed)

To fix the Patched YAML dumper for mlflow.entities.Metric warning:

# Replace the paths below with your actual project and environment paths
Copy-Item -Path "D:\YourProject\sitecustomize.py" -Destination "C:\Path\To\Your\CondaEnv\Lib\site-packages\"
Test-Path "C:\Path\To\Your\CondaEnv\Lib\site-packages\sitecustomize.py"

🔁 Example
Copy-Item -Path "D:\Final Inslit_HR\sitecustomize.py" -Destination "C:\Users\Minfy\anaconda3\envs\inslit-hr\Lib\site-packages\"
Test-Path "C:\Users\Minfy\anaconda3\envs\inslit-hr\Lib\site-packages\sitecustomize.py"


🚀 Running the Pipeline
python main.py


---

## 📈 Dataset Summary

- **Rows:** 100,000  
- **Columns:** 18  
- **Loaded From:** CSV → PostgreSQL  

### Key Columns & Observations

| Feature             | Insight                                                                 |
|---------------------|-------------------------------------------------------------------------|
| `job_title`         | Clean with 12 unique values. Check for typos and standardize.           |
| `experience_level`  | 20% missing → requires imputation.                                      |
| `employment_type`   | 24% missing → impute using mode.                                        |
| `company_size`      | Categorical with 3 values, no missing.                                  |
| `remote_ratio`      | Convert to binary feature `is_remote`.                                  |
| `base_salary`       | Negative values exist — must be cleaned.                                |
| `bonus`, `stock_options` | Used to compute `total_salary`.                                   |
| `adjusted_total_usd`| Final target variable.                                                  |
| `education`, `skills` | 100% missing — drop unless filled.                                   |

---

## ⚠️ Data Quality Issues

- Missing:
  - `experience_level` (20%)
  - `employment_type` (24%)
  - `education`, `skills` (100%) → drop
- Outliers:
  - `base_salary` has negative values
  - `salary_in_usd` max ≈ $2.3M
- Redundancy:
  - `total_salary` is derived, can be dropped
  - `salary_in_usd` vs `adjusted_total_usd` → keep only best

---

## 🧠 Feature Engineering Plan

| Feature             | Type        | Action                                              |
|---------------------|-------------|-----------------------------------------------------|
| `job_title`         | Categorical | Encode properly, group rare categories             |
| `experience_level`  | Ordinal     | Encode as Entry=0 → Exec=3                         |
| `employment_type`   | Categorical | One-hot or label encode                            |
| `company_size`      | Categorical | Encode                                              |
| `remote_ratio`      | Numerical   | Derive binary `is_remote`                          |
| `years_experience`  | Numerical   | Use as is, or bin/polynomial                       |
| `adjusted_total_usd`| Target      | Final variable to predict                          |

---

## 📊 Key Insights from EDA

- **Job Title:** Significant variation in median salaries. Strong predictor.
- **Experience Level:** Salary increases with seniority. Use ordinal mapping.
- **Company Size:** Large companies pay more on average.
- **Employment Type:** Full-time >> other types in salary.
- **Remote Ratio:** No clear pattern. `is_remote` flag is useful.
- **Years of Experience:** Moderate correlation (~0.35–0.4) with salary.

---


---

## 🧪 MLflow & SHAP Integration

- MLflow UI: `http://localhost:5000`
- All models logged with:
  - R², MAE
  - Params and artifacts
- SHAP used to explain:
  - `bonus`, `stock_options`, `years_experience`, `conversion_rate`

---

## 🔍 Drift Detection (Evidently AI)

- ✅ Drift Report: No data drift detected
- Drifted columns: 0
- Drift Report saved (HTML + JSON) and logged to MLflow
- Automatically trigger retraining if drift > `0.5`

---

## 🚀 Model Registry

- Registered model: `PricePredictor`
- Latest Production Version: `51`
- Logged and tracked using MLflow Model Registry

---

## 🛠️ Setup Instructions

1. Create Conda environment:
   ```bash
   conda create -n inslit-hr python=3.11
   conda activate inslit-hr
   pip install -r requirements.txt

Set environment variables in .env: 
DB_USER=""
DB_PASSWORD=""
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME=""
DB_TABLE_NAME=""
DB_TABLE_TEST_NAME=""
DB_TABLE_NAME_PROD=""




Run the Flask app:
python app.py



Start MLflow UI:
mlflow ui


📤 Output Logs & MLflow Tracking
✅ General Workflow Logs
Patched YAML dumper for mlflow.entities.Metric
✅ Processed data saved to PostgreSQL table: prod
🏃 View run spiffy-mare-78 at: http://localhost:5000/#/experiments/0/runs/b49287f43eb74e198ddb3d1aefc713ac
🧪 View experiment at: http://localhost:5000/#/experiments/0

✅ Top 10 Features by RF Importance
['bonus', 'company_location', 'stock_options', 'years_experience',
 'job_title', 'salary_in_usd', 'currency', 'total_salary',
 'base_salary', 'conversion_rate']



## 🤖 Model Performance Summary

| Model                   | Train R² | Test R² | Train MAE | Test MAE |
|-------------------------|----------|---------|-----------|----------|
| **XGBoost**             | 0.9179   | 0.9170  | 51470.88  | 52604.38 |
| **RandomForest**        | 0.9803   | 0.9784  | 27490.69  | 28750.86 |
| **GradientBoosting**    | 0.99995  | 0.99994 | 1153.59   | 1263.02  |
| **HistGradientBoosting**| 0.99993  | **1.0000** | 1192.55 | 1256.58 |
| LinearRegression        | -1.3084  | -1.4727 | 187156.11 | 196646.59 |
| Ridge                   | -1.3073  | -1.4714 | 187120.29 | 196609.31 |
| Lasso                   | -1.1170  | -1.2566 | 181398.60 | 190216.98 |

> 🏆 **Best Model:** `HistGradientBoosting`


🔁 Model Registration
📌 Registering model...
Model: PricePredictor
Version: 51
✅ Promoted to Production

📈 Evidently Drift Detection
HTML Report: evidently_train_vs_test_2025-07-07_10-18-51.html

JSON Report: evidently_train_vs_test_2025-07-07_10-18-51.json

🔢 Drift Metrics:
drifted_columns_count = 0.0
drifted_columns_share = 0.0
dataset_row_count = 17980
dataset_column_count = 10


✅ All drift & dataset metrics logged to MLflow.
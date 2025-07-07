# predict.py
import os
import joblib
import mlflow
import numpy as np
from data.load_data import load_data_from_postgres
from evidently_utils.evidently_logger import log_evidently_report
# Load preprocessing artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "Backend", "features", "artifacts")

label_encoders = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoders.pkl"))
scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
selected_features = joblib.load(os.path.join(ARTIFACTS_DIR, "selected_features.pkl"))

def preprocess_input(df):
    categorical_cols = label_encoders.keys()
    for col in categorical_cols:
        df[col] = df[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

    numeric_cols = ['years_experience', 'base_salary', 'bonus', 'stock_options', 'conversion_rate']
    df[numeric_cols] = scaler.transform(df[numeric_cols].astype(float))

    return df[selected_features]

def make_prediction(df):
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    os.environ["MLFLOW_ARTIFACT_URI"] = "file:///D:/Final Inslit_HR/mlruns"

    # Load Data

    # Preprocess

    df_processed = preprocess_input(df)

    # Load model from MLflow registry
    model = mlflow.pyfunc.load_model(model_uri="models:/PricePredictor/Production")
    log_preds = model.predict(df_processed)

    # Inverse log transform to get real predicted values
    preds = np.expm1(log_preds)
    return preds.tolist()

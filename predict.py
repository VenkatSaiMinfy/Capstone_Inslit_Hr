# ─────────────────────────────────────────────────────────────
# 📦 IMPORTS
# ─────────────────────────────────────────────────────────────
import os
import joblib
import mlflow
import numpy as np
from data.load_data import load_data_from_postgres               # Optional: for local testing
from evidently_utils.evidently_logger import log_evidently_report # Optional: drift reporting hook

# ─────────────────────────────────────────────────────────────
# 📁 LOAD PREPROCESSING ARTIFACTS
# ─────────────────────────────────────────────────────────────

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                      # Current directory
ARTIFACTS_DIR = os.path.join(BASE_DIR, "Backend", "features", "artifacts") # Artifacts path

# Load saved preprocessing components
label_encoders = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoders.pkl"))  # Dict of LabelEncoders
scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))                 # MinMaxScaler
selected_features = joblib.load(os.path.join(ARTIFACTS_DIR, "selected_features.pkl"))  # Top features list

# ─────────────────────────────────────────────────────────────
# 🧹 FUNCTION TO PREPROCESS INCOMING DATA BEFORE PREDICTION
# ─────────────────────────────────────────────────────────────
def preprocess_input(df):
    """
    Applies label encoding, scaling, and feature selection using saved artifacts.

    Args:
        df (pd.DataFrame): Raw input data

    Returns:
        pd.DataFrame: Processed data with only selected features
    """
    # Label encode categorical columns using loaded encoders
    categorical_cols = label_encoders.keys()
    for col in categorical_cols:
        df[col] = df[col].map(lambda x: label_encoders[col].transform([x])[0]
                              if x in label_encoders[col].classes_ else -1)

    # Normalize numeric columns using the same MinMaxScaler
    numeric_cols = ['years_experience', 'base_salary', 'bonus', 'stock_options', 'conversion_rate']
    df[numeric_cols] = scaler.transform(df[numeric_cols].astype(float))

    # Return only the selected features used during training
    return df[selected_features]

# ─────────────────────────────────────────────────────────────
# 📈 FUNCTION TO LOAD MODEL FROM MLFLOW AND MAKE PREDICTIONS
# ─────────────────────────────────────────────────────────────
def make_prediction(df):
    """
    Loads the MLflow production model and makes predictions on processed data.

    Args:
        df (pd.DataFrame): Input data

    Returns:
        list: Predicted values (USD)
    """
    # 🔗 Configure MLflow to connect to tracking server
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Local MLflow server
    os.environ["MLFLOW_ARTIFACT_URI"] = "file:///D:/Final Inslit_HR/mlruns"  # Artifact path

    # 🧹 Step 1: Preprocess the input data
    df_processed = preprocess_input(df)

    # 📦 Step 2: Load model from MLflow model registry in Production stage
    model = mlflow.pyfunc.load_model(model_uri="models:/PricePredictor/Production")

    # 🔮 Step 3: Predict the log-transformed adjusted total salary
    log_preds = model.predict(df_processed)

    # 🔁 Step 4: Apply inverse log to get actual predictions
    preds = np.expm1(log_preds)

    return preds.tolist()

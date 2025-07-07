# ─────────────────────────────────────────────────────────────
# 📦 IMPORTS
# ─────────────────────────────────────────────────────────────
from data.load_data import load_data_from_postgres                  # Load raw data from PostgreSQL
from features.preprocess import preprocess_data                    # Clean, transform, and encode raw data
from features.feature_engineering import add_feature_engineering  # (Optional) Add new features
from features.feature_selection import apply_feature_selection_rf # Select top features using RF
from training.train_model import train_and_evaluate_with_mlflow   # Model training, evaluation, logging
from analysis.eda import visualize_eda                             # EDA plots and metrics to MLflow

from CSV_TO_SQL.csv_to_sql import save_processed_to_postgres      # Save cleaned data to PostgreSQL

import mlflow
import os

# ─────────────────────────────────────────────────────────────
# 🚀 MAIN PIPELINE LOGIC
# ─────────────────────────────────────────────────────────────
def main():
    try:
        # 🔧 Set MLflow Tracking Server URI (for local or remote tracking)
        mlflow.set_tracking_uri("http://localhost:5000")

        # 🛑 End any previous MLflow run if not closed properly
        if mlflow.active_run():
            mlflow.end_run()

        # 📥 Step 1: Load data from raw PostgreSQL table
        df = load_data_from_postgres(os.getenv('DB_TABLE_NAME'))

        # 🧹 Step 2: Preprocess (clean + encode + scale)
        df, encoders, scaler = preprocess_data(df)

        # (Optional) Feature Engineering: Add new derived columns
        # df = add_feature_engineering(df)

        # 💾 Step 3: Save the cleaned data to production PostgreSQL table
        save_processed_to_postgres(df, table_name=os.getenv("DB_TABLE_NAME_PROD"))

        # 📊 Step 4: Generate EDA summary and log to MLflow
        visualize_eda(df)

        # 🛑 If visualize_eda started an MLflow run, close it
        if mlflow.active_run():
            mlflow.end_run()

        # 📁 Step 5: Set or create an MLflow experiment
        mlflow.set_experiment("USD Regression Experiment")

        # 🔁 Step 6: Start parent run to group all model comparisons
        with mlflow.start_run(run_name="All_Model_Comparisons") as parent_run:
            parent_run_id = parent_run.info.run_id

            # 🧠 Step 7: Train, tune, evaluate, and log all models
            train_and_evaluate_with_mlflow(df, parent_run_id)

    finally:
        # 🧹 Final cleanup: Ensure no run is left open
        if mlflow.active_run():
            mlflow.end_run()

# ─────────────────────────────────────────────────────────────
# 🏁 ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

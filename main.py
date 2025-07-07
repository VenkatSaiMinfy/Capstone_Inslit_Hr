from data.load_data import load_data_from_postgres
from features.preprocess import preprocess_data
from features.feature_engineering import add_feature_engineering
from features.feature_selection import apply_feature_selection_rf
from training.train_model import train_and_evaluate_with_mlflow
from analysis.eda import visualize_eda
import mlflow
import os 
from CSV_TO_SQL.csv_to_sql import save_processed_to_postgres

def main():
    try:
        mlflow.set_tracking_uri("http://localhost:5000")

        # End any previous run if exists
        if mlflow.active_run():
            mlflow.end_run()

        df = load_data_from_postgres(os.getenv('DB_TABLE_NAME'))
        df, encoders, scaler = preprocess_data(df)
        # df = add_feature_engineering(df)
        save_processed_to_postgres(df, table_name=os.getenv("DB_TABLE_NAME_PROD"))
        # visual_eda may open an mlflow run internally, so we must end it here
        visualize_eda(df)

        # Again ensure any run started inside visualize_eda is closed
        if mlflow.active_run():
            mlflow.end_run()

        mlflow.set_experiment("USD Regression Experiment")
        with mlflow.start_run(run_name="All_Model_Comparisons") as parent_run:
            parent_run_id = parent_run.info.run_id
            train_and_evaluate_with_mlflow(df, parent_run_id)

    finally:
        if mlflow.active_run():
            mlflow.end_run()

if __name__ == "__main__":
    main()

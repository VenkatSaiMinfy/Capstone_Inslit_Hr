import mlflow, shap, os
from sklearn.model_selection import GridSearchCV, train_test_split
from features.feature_selection import apply_feature_selection_rf
import numpy as np 
import os
import time
import matplotlib.pyplot as plt
import shap
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

import xgboost as xgb
from features.feature_selection import apply_feature_selection_rf

def train_and_evaluate_with_mlflow(df, parent_run_id):
    X = df.drop(columns=['adjusted_total_usd', 'adjusted_total_usd_log'])
    y = df['adjusted_total_usd_log']
    X = X.loc[:, X.std() > 1e-3]  # remove constant features

    X_selected = apply_feature_selection_rf(X, y, n_features=10)

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

    models = {
        'XGBoost': {
            'model': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1]
            }
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [5, 10]
            }
        },
        'LinearRegression': {
            'model': LinearRegression(),
            'params': {}
        }
    }

    best_model = None
    best_model_name = None
    best_r2 = -np.inf
    best_run_id = None
    client = MlflowClient()
    summary = []

    for name, cfg in models.items():
        with mlflow.start_run(run_name=name, nested=True) as run:
            print(f"\n🔧 Tuning {name}...")
            gs = GridSearchCV(cfg['model'], cfg['params'], scoring='r2', cv=5, n_jobs=-1)
            gs.fit(X_train, y_train)

            best_estimator = gs.best_estimator_
            best_params = gs.best_params_

            mlflow.log_params(best_params)

            y_train_pred_log = best_estimator.predict(X_train)
            y_test_pred_log = best_estimator.predict(X_test)
            y_train_pred = np.expm1(y_train_pred_log)
            y_test_pred = np.expm1(y_test_pred_log)
            y_train_true = np.expm1(y_train)
            y_test_true = np.expm1(y_test)

            train_mae = mean_absolute_error(y_train_true, y_train_pred)
            test_mae = mean_absolute_error(y_test_true, y_test_pred)
            train_r2 = r2_score(y_train_true, y_train_pred)
            test_r2 = r2_score(y_test_true, y_test_pred)

            mlflow.log_metrics({
                "Train_MAE": train_mae,
                "Train_R2": train_r2,
                "Test_MAE": test_mae,
                "Test_R2": test_r2
            })

            print(f"✅ {name} → Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}, Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}, Params: {best_params}")

            plt.figure(figsize=(5, 3))
            plt.bar(['Train R2', 'Test R2'], [train_r2, test_r2], color=['green', 'blue'])
            plt.title(f"{name} R2 Comparison")
            r2_plot_path = f"{name}_r2_plot.png"
            plt.savefig(r2_plot_path)
            mlflow.log_artifact(r2_plot_path)
            plt.close()
            os.remove(r2_plot_path)

            if name == "XGBoost":
                mlflow.xgboost.log_model(best_estimator, artifact_path="model")
            else:
                mlflow.sklearn.log_model(best_estimator, artifact_path="model")

            try:
                if name in ["XGBoost", "RandomForest"]:
                    print(f"📌 Generating SHAP summary plot for {name}...")
                    X_sampled = X_test[:50]
                    explainer = shap.Explainer(best_estimator, X_train)
                    shap_values = explainer(X_sampled)
                    shap.summary_plot(shap_values, X_sampled, plot_type="bar", show=False)
                    shap_path = f"{name}_shap_summary_plot.png"
                    plt.savefig(shap_path, bbox_inches="tight")
                    mlflow.log_artifact(shap_path)
                    plt.close()
                    os.remove(shap_path)
            except Exception as e:
                print(f"⚠️ SHAP failed: {e}")

            summary.append((name, train_mae, test_mae, train_r2, test_r2, best_params))

            if test_r2 > best_r2:
                best_r2 = test_r2
                best_model_name = name
                best_model = best_estimator
                best_run_id = run.info.run_id

    print("\n📈 Summary of All Models:")
    for s in summary:
        print(f"• {s[0]:15} | Train R2: {s[3]:.4f} | Test R2: {s[4]:.4f} | Test MAE: {s[2]:.2f} | Params: {s[5]}")

    print(f"\n🏆 Best Model: {best_model_name} with R2 = {best_r2:.4f}")

    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        print("\n📌 Registering & Promoting best model...")
        mv = mlflow.register_model(model_uri=model_uri, name="PricePredictor")

        for _ in range(10):
            info = client.get_model_version(name=mv.name, version=mv.version)
            if info.status == "READY":
                break
            time.sleep(1)

        client.transition_model_version_stage(
            name=mv.name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True
        )

        print(f"🚀 {mv.name} version {mv.version} → Production ✅")

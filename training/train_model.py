# ─────────────────────────────────────────────────────────────────────
# 📦 Imports required for training, logging, evaluation, and SHAP
# ─────────────────────────────────────────────────────────────────────
import os
import time
import mlflow
import shap
import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
)

from mlflow.tracking import MlflowClient
from features.feature_selection import apply_feature_selection_rf
from evidently_utils.evidently_logger import log_evidently_report

# ─────────────────────────────────────────────────────────────────────
# 🧠 Training, evaluating and registering ML models with MLflow
# ─────────────────────────────────────────────────────────────────────
def train_and_evaluate_with_mlflow(df, parent_run_id=None):
    # 🎯 Prepare target and features
    X = df.drop(columns=['adjusted_total_usd', 'adjusted_total_usd_log'])
    y = df['adjusted_total_usd_log']
    X = X.loc[:, X.std() > 1e-3]  # Remove low-variance (constant) features

    # 🌟 Apply feature selection using RandomForest
    X_selected = apply_feature_selection_rf(X, y, n_features=10)

    # 🧪 Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

    # 📚 Define models and hyperparameters
    models = {
        "XGBoost": {
            "model": xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
            "params": {"n_estimators": [50], "max_depth": [5], "learning_rate": [0.05]}
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {"n_estimators": [100], "max_depth": [5]}
        },
        "LinearRegression": {
            "model": LinearRegression(), "params": {}
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {"n_estimators": [100], "learning_rate": [0.1], "max_depth": [5]}
        },
        "HistGradientBoosting": {
            "model": HistGradientBoostingRegressor(random_state=42),
            "params": {"max_iter": [100]}
        },
        "Ridge": {
            "model": Ridge(), "params": {"alpha": [1.0, 10.0]}
        },
        "Lasso": {
            "model": Lasso(), "params": {"alpha": [0.01, 0.1]}
        }
    }

    best_r2 = -np.inf
    best_model = None
    best_model_name = None
    best_run_id = None
    summary = []

    client = MlflowClient()

    # 🔁 Train and evaluate each model
    for name, cfg in models.items():
        with mlflow.start_run(run_name=name, nested=True) as run:
            print(f"\n🔧 Tuning {name}...")

            # 🔍 Grid search with cross-validation
            gs = GridSearchCV(cfg["model"], cfg["params"], cv=5, scoring='r2', n_jobs=-1)
            gs.fit(X_train, y_train)

            best_estimator = gs.best_estimator_
            best_params = gs.best_params_

            # 📋 Log hyperparameters
            mlflow.log_params(best_params)

            # 🔮 Predict on train and test sets
            y_train_pred_log = best_estimator.predict(X_train)
            y_test_pred_log = best_estimator.predict(X_test)

            # 💸 Convert log predictions back to original scale
            y_train_pred = np.expm1(y_train_pred_log)
            y_test_pred = np.expm1(y_test_pred_log)
            y_train_true = np.expm1(y_train)
            y_test_true = np.expm1(y_test)

            # 📊 Evaluate performance
            metrics = {
                "Train_MAE": mean_absolute_error(y_train_true, y_train_pred),
                "Test_MAE": mean_absolute_error(y_test_true, y_test_pred),
                "Train_R2": r2_score(y_train_true, y_train_pred),
                "Test_R2": r2_score(y_test_true, y_test_pred),
                "Train_RMSE": np.sqrt(mean_squared_error(y_train_true, y_train_pred)),
                "Test_RMSE": np.sqrt(mean_squared_error(y_test_true, y_test_pred)),
                "Train_MAPE": mean_absolute_percentage_error(y_train_true, y_train_pred),
                "Test_MAPE": mean_absolute_percentage_error(y_test_true, y_test_pred),
            }

            mlflow.log_metrics(metrics)

            print(f"✅ {name} → Train R2: {metrics['Train_R2']:.6f}, Test R2: {metrics['Test_R2']:.6f}, "
                  f"Train MAE: {metrics['Train_MAE']:.2f}, Test MAE: {metrics['Test_MAE']:.2f}")

            # 📈 Plot R2 bar chart
            plt.figure(figsize=(6, 3))
            plt.bar(["Train R2", "Test R2"], [metrics["Train_R2"], metrics["Test_R2"]], color=["green", "blue"])
            plt.title(f"{name} R2")
            plot_path = f"{name}_r2_plot.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            plt.close()
            os.remove(plot_path)

            # 💾 Log model
            if name == "XGBoost":
                mlflow.xgboost.log_model(best_estimator, artifact_path="model")
            else:
                mlflow.sklearn.log_model(best_estimator, artifact_path="model")

            # 🔍 Explain with SHAP (if supported)
            try:
                if name in ["XGBoost", "RandomForest", "GradientBoosting"]:
                    print(f"📌 SHAP for {name}...")
                    explainer = shap.Explainer(best_estimator, X_train)
                    shap_values = explainer(X_test[:50])
                    shap.summary_plot(shap_values, X_test[:50], plot_type="bar", show=False)
                    shap_path = f"{name}_shap.png"
                    plt.savefig(shap_path, bbox_inches="tight")
                    mlflow.log_artifact(shap_path)
                    plt.close()
                    os.remove(shap_path)
            except Exception as e:
                print(f"⚠️ SHAP failed: {e}")

            # 📦 Track summary
            summary.append((name, metrics["Train_R2"], metrics["Test_R2"],
                            metrics["Train_MAE"], metrics["Test_MAE"], best_params))

            # 🏆 Update best model if higher R2
            if metrics["Test_R2"] > best_r2:
                best_r2 = metrics["Test_R2"]
                best_model = best_estimator
                best_model_name = name
                best_run_id = run.info.run_id

    # 📊 Print summary of all models
    print("\n📊 Summary of All Models:")
    for s in summary:
        print(f"• {s[0]:20} | Train R2: {s[1]:.4f} | Test R2: {s[2]:.4f} | "
              f"Train MAE: {s[3]:.2f} | Test MAE: {s[4]:.2f} | Params: {s[5]}")

    print(f"\n🏆 Best Model: {best_model_name} → Test R2 = {best_r2:.4f}")

    # 📌 Register the best model and promote to Production
    if best_run_id:
        print("\n📌 Registering model...")
        mv = mlflow.register_model(f"runs:/{best_run_id}/model", name="PricePredictor")

        # Wait until model is ready
        for _ in range(10):
            info = client.get_model_version(name=mv.name, version=mv.version)
            if info.status == "READY":
                break
            time.sleep(1)

        # Promote to Production stage and archive older versions
        client.transition_model_version_stage(
            name=mv.name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"🚀 Model '{mv.name}' version {mv.version} promoted to Production ✅")

        # 🧪 Generate train vs test drift report
        print("\n📈 Logging Evidently drift report...")
        log_evidently_report(X_train, X_test, dataset_name="train_vs_test")

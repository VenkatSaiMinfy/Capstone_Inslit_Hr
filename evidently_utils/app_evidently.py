# 📦 Required imports
import json
from pathlib import Path
from datetime import datetime

import mlflow                          # For experiment tracking
from evidently import Report           # Core class for generating reports
from evidently.presets import DataDriftPreset  # Preset to detect data drift

# 🏷️ Columns that should be treated as strings even if they contain numbers
LABEL_LIKE_COLUMNS = ['skills', 'job_title', 'education_level']

def log_evidently_report(reference_data, current_data, dataset_name="train_vs_test"):
    """
    🚨 Generate and log an Evidently data drift report, save HTML and JSON versions,
    and log drift-related metrics to MLflow.

    Args:
        reference_data (DataFrame): Reference dataset (typically training data)
        current_data (DataFrame): Current or new dataset (e.g. test or batch data)
        dataset_name (str): Custom label to identify this report in filenames and metrics
    """

    # 🔍 Keep only common columns between both datasets
    common_cols = set(reference_data.columns).intersection(current_data.columns)
    if not common_cols:
        print(f"⚠️ No common columns between reference and {dataset_name}; skipping Evidently report.")
        return

    # 📐 Create clean DataFrames with only common columns
    ref = reference_data[list(sorted(common_cols))].copy()
    cur = current_data[list(sorted(common_cols))].copy()

    # 🧹 Ensure column names are strings
    ref.columns = ref.columns.map(str)
    cur.columns = cur.columns.map(str)

    # 🚫 Remove columns that are completely empty
    ref.dropna(axis=1, how='all', inplace=True)
    cur.dropna(axis=1, how='all', inplace=True)

    # 🔤 Convert label-like categorical columns to string to avoid float misclassification
    for col in LABEL_LIKE_COLUMNS:
        if col in ref.columns:
            ref[col] = ref[col].astype(str)
        if col in cur.columns:
            cur[col] = cur[col].astype(str)

    # 📊 Run the data drift preset (only this preset to avoid errors on label columns)
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=ref, current_data=cur)

    # 📁 Prepare paths for saving report artifacts
    save_dir = Path.cwd() / "evidently_reports"
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    html_path = save_dir / f"evidently_{dataset_name}_{ts}.html"
    json_path = save_dir / f"evidently_{dataset_name}_{ts}.json"

    # 💾 Save HTML and JSON versions of the report
    result.save_html(str(html_path))
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(result.json())

    # 📥 Log both artifacts to MLflow under `evidently/` folder
    mlflow.log_artifact(str(html_path), artifact_path="evidently")
    mlflow.log_artifact(str(json_path), artifact_path="evidently")
    print(f"📄 Logged HTML: {html_path.name}")
    print(f"🗄️ Logged JSON: {json_path.name}")

    # 📊 Load the JSON report to extract metrics
    report_json = json.loads(json_path.read_text(encoding='utf-8'))
    metrics_list = report_json.get("metrics", [])

    # 🔢 Log summary drift metrics: count and share of drifted columns
    drift_entry = next((m for m in metrics_list if m.get("metric_id", "").startswith("DriftedColumnsCount")), None)
    if drift_entry:
        count = drift_entry["value"]["count"]
        share = drift_entry["value"]["share"]
        mlflow.log_metric(f"{dataset_name}__drifted_columns_count", float(count))
        mlflow.log_metric(f"{dataset_name}__drifted_columns_share", float(share))
        print(f"🔢 {dataset_name}__drifted_columns_count = {count}")
        print(f"🔢 {dataset_name}__drifted_columns_share = {share}")
    else:
        print("⚠️ No DriftedColumnsCount entry found.")

    # 🧬 Log per-feature drift values to MLflow
    for m in metrics_list:
        mid = m.get("metric_id", "")
        if mid.startswith("ValueDrift(column="):
            col = mid.split("=")[1].rstrip(")")
            val = m.get("value")
            if isinstance(val, (int, float)):
                mlflow.log_metric(f"{dataset_name}__drift_{col}", float(val))
                print(f"🔢 {dataset_name}__drift_{col} = {val}")

    # ✅ Done logging all drift-related insights
    print(f"✅ All drift metrics for `{dataset_name}` logged to MLflow.")

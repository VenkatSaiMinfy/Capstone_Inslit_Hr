import json
from pathlib import Path
from datetime import datetime

import mlflow
from evidently import Report
from evidently.presets import DataDriftPreset

# List any categorical label-like columns that may contain numeric values
LABEL_LIKE_COLUMNS = ['skills', 'job_title', 'education_level']

def log_evidently_report(reference_data, current_data, dataset_name="train_vs_test"):
    """
    Generates an Evidently data drift report on the common columns between reference and current,
    saves HTML and JSON, logs both as MLflow artifacts, and extracts key drift metrics including per-feature drift.

    Args:
        reference_data: DataFrame for reference (e.g. training) dataset.
        current_data: DataFrame for current (e.g. test or new batch) dataset.
        dataset_name: Identifier used for naming artifacts and metric prefixes.
    """
    # Align and prepare columns
    common_cols = set(reference_data.columns).intersection(current_data.columns)
    if not common_cols:
        print(f"⚠️ No common columns between reference and {dataset_name}; skipping Evidently report.")
        return

    # Subset to common columns
    ref = reference_data[list(sorted(common_cols))].copy()
    cur = current_data[list(sorted(common_cols))].copy()

    # Ensure column names are strings
    ref.columns = ref.columns.map(str)
    cur.columns = cur.columns.map(str)

    # Drop fully empty columns
    ref.dropna(axis=1, how='all', inplace=True)
    cur.dropna(axis=1, how='all', inplace=True)

    # Cast label-like columns to string to avoid float labels
    for col in LABEL_LIKE_COLUMNS:
        if col in ref.columns:
            ref[col] = ref[col].astype(str)
        if col in cur.columns:
            cur[col] = cur[col].astype(str)

    # Run only drift preset to avoid float label errors
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=ref, current_data=cur)

    # Save artifacts
    save_dir = Path.cwd() / "evidently_reports"
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    html_path = save_dir / f"evidently_{dataset_name}_{ts}.html"
    json_path = save_dir / f"evidently_{dataset_name}_{ts}.json"

    result.save_html(str(html_path))
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(result.json())

    # Log to MLflow
    mlflow.log_artifact(str(html_path), artifact_path="evidently")
    mlflow.log_artifact(str(json_path), artifact_path="evidently")
    print(f"📄 Logged HTML: {html_path.name}")
    print(f"🗄️ Logged JSON: {json_path.name}")

    # Extract metrics
    report_json = json.loads(json_path.read_text(encoding='utf-8'))
    metrics_list = report_json.get("metrics", [])

    # Overall drifted columns metrics
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

    # Per-feature drift
    for m in metrics_list:
        mid = m.get("metric_id", "")
        if mid.startswith("ValueDrift(column="):
            col = mid.split("=")[1].rstrip(")")
            val = m.get("value")
            if isinstance(val, (int, float)):
                mlflow.log_metric(f"{dataset_name}__drift_{col}", float(val))
                print(f"🔢 {dataset_name}__drift_{col} = {val}")

    print(f"✅ All drift metrics for `{dataset_name}` logged to MLflow.")

import json
from pathlib import Path
from datetime import datetime

import mlflow
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

def log_evidently_report(reference_data, current_data, dataset_name="train_vs_test"):
    """
    Generates an Evidently data drift + summary report on the common columns between reference and current,
    saves HTML and JSON, logs both as MLflow artifacts, and extracts key drift metrics including per-feature drift.

    Args:
        reference_data: DataFrame for reference (e.g. training) dataset.
        current_data: DataFrame for current (e.g. test or new batch) dataset.
        dataset_name: Identifier used for naming artifacts and metric prefixes.
    """
    # 0️⃣ Align columns: use only the intersection to avoid partial-column errors
    common_cols = set(reference_data.columns).intersection(current_data.columns)
    if not common_cols:
        print(f"⚠️ No common columns between reference and {dataset_name}; skipping Evidently report.")
        return
    ref = reference_data[sorted(common_cols)]
    cur = current_data[sorted(common_cols)]

    # 1️⃣ Run the Evidently report (drift + summary)
    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    result=report.run(reference_data=ref, current_data=cur)

    # 2️⃣ Ensure local save directory exists
    save_dir = Path.cwd() / "evidently_reports"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 3️⃣ Save HTML and JSON
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    html_path = save_dir / f"evidently_{dataset_name}_{ts}.html"
    json_path = save_dir / f"evidently_{dataset_name}_{ts}.json"

    result.save_html(str(html_path))


    with open(json_path, "w", encoding="utf-8") as f:
        f.write(result.json())

    # 4️⃣ Log artifacts to MLflow
    mlflow.log_artifact(str(html_path), artifact_path="evidently")
    mlflow.log_artifact(str(json_path), artifact_path="evidently")
    print(f"📄 Logged HTML: {html_path.name}")
    print(f"🗄️  Logged JSON: {json_path.name}")

    # 5️⃣ Load JSON and extract metrics list
    with open(json_path, "r", encoding="utf-8") as fp:
        report_json = json.load(fp)
    metrics_list = report_json.get("metrics", [])

    # 6️⃣ Overall drifted columns metrics
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

    # 7️⃣ Row and column counts
    rowcount = next((m["value"] for m in metrics_list if m.get("metric_id") == "RowCount()"), None)
    colcount = next((m["value"] for m in metrics_list if m.get("metric_id") == "ColumnCount()"), None)
    if rowcount is not None:
        mlflow.log_metric(f"{dataset_name}__dataset_row_count", float(rowcount))
        print(f"🔢 {dataset_name}__dataset_row_count = {rowcount}")
    if colcount is not None:
        mlflow.log_metric(f"{dataset_name}__dataset_column_count", float(colcount))
        print(f"🔢 {dataset_name}__dataset_column_count = {colcount}")

    # 8️⃣ Per-feature value drift metrics
    for m in metrics_list:
        mid = m.get("metric_id", "")
        if mid.startswith("ValueDrift(column="):
            # extract column name
            col = mid.split("=")[1].rstrip(")")
            val = m.get("value")
            if isinstance(val, (int, float)):
                mlflow.log_metric(f"{dataset_name}__drift_{col}", float(val))
                print(f"🔢 {dataset_name}__drift_{col} = {val}")

    print(f"✅ All drift & dataset metrics for `{dataset_name}` logged to MLflow.")
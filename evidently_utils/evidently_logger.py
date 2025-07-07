# 📦 Import necessary libraries
import json
from pathlib import Path
from datetime import datetime

import mlflow                          # MLflow for tracking experiments
from evidently import Report           # Core class to generate reports
from evidently.presets import DataDriftPreset, DataSummaryPreset  # Built-in metric presets

def log_evidently_report(reference_data, current_data, dataset_name="train_vs_test"):
    """
    📊 Generates an Evidently report (data drift + summary), saves both HTML and JSON formats,
    logs them as MLflow artifacts, and extracts key drift metrics including per-column drift.

    Parameters:
    ----------
    reference_data : pd.DataFrame
        The reference dataset (typically training or historical data).
    current_data : pd.DataFrame
        The current dataset (e.g. test, production batch, or new data).
    dataset_name : str
        Identifier used to prefix saved files and metric names.
    """

    # 0️⃣ Align datasets by selecting only the common columns
    common_cols = set(reference_data.columns).intersection(current_data.columns)
    if not common_cols:
        print(f"⚠️ No common columns between reference and {dataset_name}; skipping Evidently report.")
        return
    
    ref = reference_data[sorted(common_cols)]  # Ensure column order is consistent
    cur = current_data[sorted(common_cols)]

    # 1️⃣ Create an Evidently report using both DataDrift and DataSummary presets
    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    result = report.run(reference_data=ref, current_data=cur)

    # 2️⃣ Create a directory to save reports (if not already exists)
    save_dir = Path.cwd() / "evidently_reports"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 3️⃣ Save reports in HTML and JSON formats with timestamps
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    html_path = save_dir / f"evidently_{dataset_name}_{ts}.html"
    json_path = save_dir / f"evidently_{dataset_name}_{ts}.json"

    result.save_html(str(html_path))  # Save HTML report
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(result.json())        # Save JSON report

    # 4️⃣ Log both report files as MLflow artifacts under "evidently/" folder
    mlflow.log_artifact(str(html_path), artifact_path="evidently")
    mlflow.log_artifact(str(json_path), artifact_path="evidently")
    print(f"📄 Logged HTML: {html_path.name}")
    print(f"🗄️  Logged JSON: {json_path.name}")

    # 5️⃣ Load the JSON report content to extract metrics
    with open(json_path, "r", encoding="utf-8") as fp:
        report_json = json.load(fp)
    metrics_list = report_json.get("metrics", [])

    # 6️⃣ Extract and log overall drift metrics
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

    # 7️⃣ Log overall row and column counts from DataSummary
    rowcount = next((m["value"] for m in metrics_list if m.get("metric_id") == "RowCount()"), None)
    colcount = next((m["value"] for m in metrics_list if m.get("metric_id") == "ColumnCount()"), None)

    if rowcount is not None:
        mlflow.log_metric(f"{dataset_name}__dataset_row_count", float(rowcount))
        print(f"🔢 {dataset_name}__dataset_row_count = {rowcount}")
    if colcount is not None:
        mlflow.log_metric(f"{dataset_name}__dataset_column_count", float(colcount))
        print(f"🔢 {dataset_name}__dataset_column_count = {colcount}")

    # 8️⃣ Log individual feature drift values (only numerical drift values)
    for m in metrics_list:
        mid = m.get("metric_id", "")
        if mid.startswith("ValueDrift(column="):
            # Extract the column name from the metric ID
            col = mid.split("=")[1].rstrip(")")
            val = m.get("value")
            if isinstance(val, (int, float)):
                mlflow.log_metric(f"{dataset_name}__drift_{col}", float(val))
                print(f"🔢 {dataset_name}__drift_{col} = {val}")

    # ✅ Done logging all relevant metrics and artifacts
    print(f"✅ All drift & dataset metrics for `{dataset_name}` logged to MLflow.")

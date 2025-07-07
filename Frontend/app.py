from flask import Flask, render_template, request
import os
import sys
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

# Add backend path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import backend modules
from CSV_TO_SQL.csv_to_sql import save_csv_to_postgres
from data.load_data import load_data_from_postgres
from predict import make_prediction
from evidently_utils.app_evidently import log_evidently_report

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')
import time  # ⏱️ Add this at the top of your file

@app.route('/upload', methods=['POST'])
def upload():
    start_time = time.time()

    file = request.files['file']
    filename = file.filename
    table_name = os.path.splitext(filename)[0].strip().lower()

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    print(f"[{filename}] ✅ Saved file in {time.time() - start_time:.2f} sec")

    # Step 1: Save to PostgreSQL
    t1 = time.time()
    save_csv_to_postgres(file_path, table_name)
    print(f"[{filename}] ✅ Saved to PostgreSQL in {time.time() - t1:.2f} sec")

    # Step 2: Load uploaded data
    t2 = time.time()
    current_df = load_data_from_postgres(table_name)
    print(f"[{filename}] ✅ Loaded current batch from DB in {time.time() - t2:.2f} sec")

    # Step 3: Load reference data
    t3 = time.time()
    reference_table = os.getenv("DB_TABLE_NAME_PROD")
    reference_df = load_data_from_postgres(reference_table)
    print(f"[{filename}] ✅ Loaded reference data in {time.time() - t3:.2f} sec")

    # Step 5: Make predictions
    t5 = time.time()
    preds = make_prediction(current_df)
    current_df['Predictions'] = preds
    print(f"[{filename}] ✅ Predictions completed in {time.time() - t5:.2f} sec")

    # Step 4: Evidently report
    t4 = time.time()
    log_evidently_report(reference_df, current_df, dataset_name=table_name)
    print(f"[{filename}] ✅ Generated Evidently report in {time.time() - t4:.2f} sec")

    # Final time
    print(f"[{filename}] 🚀 Total time taken: {time.time() - start_time:.2f} sec")

    column_names = current_df.columns.tolist()
    data = current_df.values.tolist()
    return render_template('results.html', column_names=column_names, data=data)


if __name__ == '__main__':
    app.run(debug=True, port=5001)

from flask import Flask, render_template, request, send_file
import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Add the root project directory to Python path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import your backend modules
from CSV_TO_SQL.csv_to_sql import save_csv_to_postgres
from data.load_data import load_data_from_postgres
from predict import make_prediction

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Step 1: Save CSV to PostgreSQL
    save_csv_to_postgres(file_path)

    # Step 2: Load the data back from PostgreSQL
    df = load_data_from_postgres()

    # Step 3: Predict
    preds = make_prediction(df)
    df['Predictions'] = preds

    # Step 4: Display results
    column_names = df.columns.tolist()
    data = df.values.tolist()
    return render_template('results.html', column_names=column_names, data=data)

if __name__ == '__main__':
    app.run(debug=True,port=5001)

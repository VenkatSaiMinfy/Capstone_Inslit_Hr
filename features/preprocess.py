import os
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np

def preprocess_data(df):
    """
    🧹 Preprocess the raw salary dataset:
    - Clean and filter data
    - Encode categorical features
    - Normalize numeric features
    - Add new features
    - Save label encoders and scaler for later use

    Parameters:
    -----------
    df : pd.DataFrame
        Raw input data

    Returns:
    --------
    df : pd.DataFrame
        Preprocessed data
    label_encoders : dict
        Dictionary of LabelEncoders for categorical columns
    scaler : MinMaxScaler
        Fitted MinMaxScaler for numeric columns
    """

    # 1️⃣ Drop unnecessary columns if they exist
    df = df.drop(columns=['education', 'skills'], errors='ignore')

    # 2️⃣ Remove rows with invalid (negative) values in salary-related fields
    df = df[
        (df['base_salary'] >= 0) &
        (df['bonus'] >= 0) &
        (df['stock_options'] >= 0) &
        (df['adjusted_total_usd'] >= 0)
    ]

    # 3️⃣ Drop any duplicate rows
    df = df.drop_duplicates()

    # 4️⃣ Fill missing values in categorical columns using mode (most frequent value)
    df.loc[:, 'experience_level'] = df['experience_level'].fillna(df['experience_level'].mode()[0])
    df.loc[:, 'employment_type'] = df['employment_type'].fillna(df['employment_type'].mode()[0])

    # 5️⃣ Create binary column: 1 if remote_ratio is 100%, else 0
    df.loc[:, 'is_remote'] = df['remote_ratio'].apply(lambda x: 1 if x == 100 else 0)

    # 6️⃣ Create a new feature: total_salary = base + bonus + stock options
    df.loc[:, 'total_salary'] = df['base_salary'] + df['bonus'] + df['stock_options']

    # 7️⃣ Encode categorical columns using LabelEncoder
    categorical_cols = [
        'job_title', 'experience_level', 'currency',
        'employment_type', 'company_size', 'company_location', 'salary_currency'
    ]
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col]).astype(int)  # Ensure encoded values are integers
        label_encoders[col] = le

    # 8️⃣ Normalize numeric columns using MinMaxScaler
    numeric_cols = [
        'years_experience', 'base_salary', 'bonus',
        'stock_options', 'conversion_rate'
    ]
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols].astype(float))

    # 9️⃣ Log transform the target variable for better modeling (smoother distribution)
    df.loc[:, 'adjusted_total_usd_log'] = np.log1p(df['adjusted_total_usd'])

    # 🔟 Save the fitted encoders and scaler to disk for reuse during inference
    os.makedirs("Backend/features/artifacts", exist_ok=True)
    joblib.dump(label_encoders, "Backend/features/artifacts/label_encoders.pkl")
    joblib.dump(scaler, "Backend/features/artifacts/scaler.pkl")

    # ✅ Return processed DataFrame and transformation objects
    return df, label_encoders, scaler

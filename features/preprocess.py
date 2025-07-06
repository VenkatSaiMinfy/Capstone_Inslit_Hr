import os
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
'''
def preprocess_data(df):
    df = df.drop(columns=['education', 'skills'], errors='ignore')
    df = df[(df['base_salary'] >= 0) & (df['bonus'] >= 0) & (df['stock_options'] >= 0) & (df['adjusted_total_usd'] >= 0)]
    df = df.drop_duplicates()

    df['experience_level'] = df['experience_level'].fillna(df['experience_level'].mode()[0])
    df['employment_type'] = df['employment_type'].fillna(df['employment_type'].mode()[0])
    df['is_remote'] = df['remote_ratio'].apply(lambda x: 1 if x == 100 else 0)
    df['total_salary'] = df['base_salary'] + df['bonus'] + df['stock_options']

    categorical_cols = ['job_title', 'experience_level', 'currency', 'employment_type', 'company_size', 'company_location', 'salary_currency']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    numeric_cols = ['years_experience', 'base_salary', 'bonus', 'stock_options', 'conversion_rate']
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols].astype(float))

    df['adjusted_total_usd_log'] = np.log1p(df['adjusted_total_usd'])

    # Save encoders and scaler
    os.makedirs("Backend/features/artifacts", exist_ok=True)
    joblib.dump(label_encoders, "Backend/features/artifacts/label_encoders.pkl")
    joblib.dump(scaler, "Backend/features/artifacts/scaler.pkl")

    return df, label_encoders, scaler
'''


from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np

def preprocess_data(df):
    # Drop unnecessary columns if they exist
    df = df.drop(columns=['education', 'skills'], errors='ignore')

    # Filter out rows with negative salary-related values
    df = df[
        (df['base_salary'] >= 0) &
        (df['bonus'] >= 0) &
        (df['stock_options'] >= 0) &
        (df['adjusted_total_usd'] >= 0)
    ]

    # Drop duplicates
    df = df.drop_duplicates()

    # Fill missing categorical values
    df.loc[:, 'experience_level'] = df['experience_level'].fillna(df['experience_level'].mode()[0])
    df.loc[:, 'employment_type'] = df['employment_type'].fillna(df['employment_type'].mode()[0])

    # Add binary feature for remote jobs
    df.loc[:, 'is_remote'] = df['remote_ratio'].apply(lambda x: 1 if x == 100 else 0)

    # Calculate total salary
    df.loc[:, 'total_salary'] = df['base_salary'] + df['bonus'] + df['stock_options']

    # Encode categorical columns
    categorical_cols = [
        'job_title', 'experience_level', 'currency',
        'employment_type', 'company_size', 'company_location', 'salary_currency'
    ]
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col]).astype(int)  # cast to int
        label_encoders[col] = le


    # Normalize numeric columns
    numeric_cols = [
        'years_experience', 'base_salary', 'bonus',
        'stock_options', 'conversion_rate'
    ]
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols].astype(float))

    # Add log-transformed target variable
    df.loc[:, 'adjusted_total_usd_log'] = np.log1p(df['adjusted_total_usd'])
    # Save encoders and scaler
    os.makedirs("Backend/features/artifacts", exist_ok=True)
    joblib.dump(label_encoders, "Backend/features/artifacts/label_encoders.pkl")
    joblib.dump(scaler, "Backend/features/artifacts/scaler.pkl")
    return df, label_encoders, scaler
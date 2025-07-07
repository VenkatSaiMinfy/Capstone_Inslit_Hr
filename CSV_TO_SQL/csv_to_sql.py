import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

def save_csv_to_postgres(csv_path, table_name):
    df = pd.read_csv(csv_path)
    engine = create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"✅ Saved {csv_path} to table: {table_name}")
    return df

def save_processed_to_postgres(df, table_name="processed_data"):
    engine = create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"✅ Processed data saved to PostgreSQL table: {table_name}")

# ✅ Wrap this in a main guard
if __name__ == "__main__":
    save_csv_to_postgres('../Dataset_csv/Software_Salaries.csv', os.getenv('DB_TABLE_NAME'))

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv()
import os
# 1. Load the CSV file
csv_file_path = "Dataset_csv/Software_Salaries.csv"  
df = pd.read_csv(csv_file_path)

# 2. PostgreSQL connection details
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_TABLE=os.getenv('DB_TABLE_NAME')


# 3. Create a connection engine
engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# 4. Load the DataFrame to PostgreSQL
df.to_sql(DB_TABLE, engine, if_exists='replace', index=False)

print(f"✅ Successfully loaded {len(df)} records into the '{DB_TABLE}' table.")

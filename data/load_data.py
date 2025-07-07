# 📦 Import required libraries
import pandas as pd                          # For working with DataFrames
from sqlalchemy import create_engine         # For connecting to PostgreSQL
import os                                    # To access environment variables
from dotenv import load_dotenv               # To load variables from .env file

# 📥 Load environment variables from .env file
load_dotenv()

# 📡 Function to load data from a PostgreSQL table into a Pandas DataFrame
def load_data_from_postgres(table_name):
    # 🔧 Create a SQLAlchemy engine using PostgreSQL connection string
    engine = create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

    # 🧾 Read data from the given PostgreSQL table and return as DataFrame
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)

# 📦 Import required libraries
import pandas as pd                          # For reading CSV and handling DataFrames
from sqlalchemy import create_engine         # For connecting to PostgreSQL
from dotenv import load_dotenv               # To load environment variables from .env file
import os                                    # To access OS environment variables

# 📥 Load environment variables from the .env file
load_dotenv()

# 🚀 Function to load CSV data into a specified PostgreSQL table
def save_csv_to_postgres(csv_path, table_name):
    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Create SQLAlchemy engine using PostgreSQL connection string
    engine = create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

    # Save the DataFrame to the specified PostgreSQL table (overwrite if exists)
    df.to_sql(table_name, engine, if_exists='replace', index=False)

    print(f"✅ Saved {csv_path} to table: {table_name}")
    return df  # Return the DataFrame for optional reuse

# 💾 Function to save a processed DataFrame to PostgreSQL
def save_processed_to_postgres(df, table_name="processed_data"):
    # Create SQLAlchemy engine again for reuse
    engine = create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

    # Save processed data into the specified PostgreSQL table
    df.to_sql(table_name, engine, if_exists='replace', index=False)

    print(f"✅ Processed data saved to PostgreSQL table: {table_name}")

# 🛡️ Execute only if the script is run directly
if __name__ == "__main__":
    # Call the function with a specific CSV and table name from environment variables
    save_csv_to_postgres('../Dataset_csv/Software_Salaries.csv', os.getenv('DB_TABLE_NAME'))

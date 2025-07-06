from data.load_data import load_data_from_postgres
from features.preprocess import preprocess_data
from features.feature_engineering import add_feature_engineering
from features.feature_selection import load_selected_features
from predict import make_prediction

def run_pipeline():
    # Load raw data
    df = load_data_from_postgres()

    # Preprocess returns df, encoders, scaler — unpack it
    df, _, _ = preprocess_data(df)

    # Feature Selection
    df = load_selected_features(df)

    # Prediction
    predictions = make_prediction(df)

    df['Predictions'] = predictions
    return df

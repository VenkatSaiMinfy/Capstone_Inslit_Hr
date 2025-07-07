# 📦 Import necessary libraries
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib
import os

def apply_feature_selection_rf(X, y, n_features=14):
    """
    📊 Select top `n_features` based on feature importance using RandomForestRegressor.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix (independent variables)
    y : pd.Series or array
        Target variable (dependent variable)
    n_features : int
        Number of top features to select

    Returns:
    --------
    pd.DataFrame
        Subset of X with selected important features
    """

    # 1️⃣ Initialize and train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 2️⃣ Get feature importances and select top N features
    importances = model.feature_importances_                          # Get importance scores for all features
    indices = np.argsort(importances)[-n_features:]                  # Indices of top `n_features`
    selected_columns = X.columns[indices]                            # Get actual column names from indices

    # 3️⃣ Print selected features to the console
    print(f"\n✅ Top {n_features} features by RF importance:\n", list(selected_columns))

    # 4️⃣ Save selected feature names to a file for future use (e.g., during prediction)
    os.makedirs("Backend/features/artifacts", exist_ok=True)         # Ensure the output directory exists
    joblib.dump(list(selected_columns), "Backend/features/artifacts/selected_features.pkl")

    # 5️⃣ Return filtered DataFrame with only the selected features
    return X[selected_columns]

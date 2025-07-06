# Backend/features/feature_selection.py
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib
import os
'''
def apply_feature_selection_rf(X, y, n_features=10):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[-n_features:]
    selected_columns = X.columns[indices]

    print(f"\n✅ Top {n_features} features by RF importance:\n", list(selected_columns))

    os.makedirs("Backend/features/artifacts", exist_ok=True)
    joblib.dump(list(selected_columns), "Backend/features/artifacts/selected_features.pkl")

    return X[selected_columns]'''
    
def apply_feature_selection_rf(X, y, n_features=14):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[-n_features:]
    selected_columns = X.columns[indices]
    print(f"\n✅ Top {n_features} features by RF importance:\n", list(selected_columns))
    os.makedirs("Backend/features/artifacts", exist_ok=True)
    joblib.dump(list(selected_columns), "Backend/features/artifacts/selected_features.pkl")
    return X[selected_columns]
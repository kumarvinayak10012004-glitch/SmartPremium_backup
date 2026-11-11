
import pandas as pd
import joblib
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error

from src.preprocessing import preprocess_train


# 1. Load data
data_path = "data/playground-series-s4e12/train.csv"
df = pd.read_csv(data_path)

print("Dataset loaded successfully!")


# 2. Preprocess data
X, y, encoder, num_imputer, cat_imputer = preprocess_train(
    df,
    target_col="Premium Amount"
)


# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Define four regression models
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
}


# 5. Train and evaluate all models
results = {}

best_rmse = float("inf")
best_model = None
best_model_name = None

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    results[name] = rmse

    print(f"{name} RMSE: {rmse:.4f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_model_name = name


# 6. Model comparison
print("\n========== MODEL COMPARISON ==========")

for name, rmse in results.items():
    print(f"{name}: RMSE = {rmse:.4f}")

print("\nBest Model:", best_model_name)
print("Best RMSE:", best_rmse)


# 7. Save best model and preprocessing files
os.makedirs("artifacts", exist_ok=True)

joblib.dump(best_model, "artifacts/model.pkl")
joblib.dump(encoder, "artifacts/encoder.pkl")
joblib.dump(num_imputer, "artifacts/num_imputer.pkl")
joblib.dump(cat_imputer, "artifacts/cat_imputer.pkl")
joblib.dump(results, "artifacts/model_results.pkl")

print("\nBest model, encoder and imputers saved successfully!")

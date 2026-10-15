
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset and the saved model
df = pd.read_csv("train.csv")
model = joblib.load("real_estate_model.pkl")

# 2. Recreate the SAME features used during training
df["PropertyAge"] = df["YrSold"] - df["YearBuilt"]
df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

df["TotalSF"] = (
    df["TotalBsmtSF"].fillna(0)
    + df["1stFlrSF"].fillna(0)
    + df["2ndFlrSF"].fillna(0)
)

df["TotalBathrooms"] = (
    df["FullBath"].fillna(0)
    + 0.5 * df["HalfBath"].fillna(0)
)

df["TotalPorchSF"] = (
    df["OpenPorchSF"].fillna(0)
    + df["3SsnPorch"].fillna(0)
    + df["EnclosedPorch"].fillna(0)
    + df["ScreenPorch"].fillna(0)
    + df["WoodDeckSF"].fillna(0)
)

# 3. Separate features and actual prices
X = df.drop(columns=["SalePrice", "Id"])
y = df["SalePrice"]

# 4. Recreate the same 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 5. Predict using the SAVED model (do not retrain it)
predictions = model.predict(X_test)

# 6. Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n===== SAVED MODEL: TEST RESULTS =====")
print(f"Test samples: {len(X_test)}")
print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²:    {r2:.4f}")

# 7. Compare actual vs predicted prices
results = pd.DataFrame({
    "ActualPrice": y_test.values,
    "PredictedPrice": predictions,
    "AbsoluteError": np.abs(y_test.values - predictions)
})

print("\n===== FIRST 15 TEST PREDICTIONS =====")
print(results.head(15).round(2).to_string(index=False))

# 8. Save all test predictions for inspection
results.to_csv("test_predictions.csv", index=False)
print("\nSaved comparisons to test_predictions.csv")

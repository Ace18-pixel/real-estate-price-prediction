import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# 1. LOAD DATA

df = pd.read_csv("train.csv")

print("Dataset shape:", df.shape)
print(df.head())


# 2. FEATURE ENGINEERING

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

print("\nFeature engineering completed.")

print(df[[
    "PropertyAge",
    "RemodAge",
    "TotalSF",
    "TotalBathrooms",
    "TotalPorchSF"
]].head())


# 3. FEATURES AND TARGET

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Remove ID because it is only an identifier
X = X.drop("Id", axis=1)

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# 4. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# 5. PREPROCESSING

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns

print("\nNumerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# Numerical preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])


# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Combine numerical + categorical preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("Preprocessing pipeline created.")


# 6. GRADIENT BOOSTING MODEL

gradient_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])

print("\nTraining Gradient Boosting model...")

gradient_model.fit(X_train, y_train)

print("Model training completed.")


# 7. MODEL EVALUATION

print("\nMaking predictions on test data...")

y_pred = gradient_model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\n===== FINAL MODEL RESULTS =====")
print(f"MAE : ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²  : {r2:.4f}")


# 8. ACTUAL VS PREDICTED


plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.6)

# Perfect prediction line
min_price = min(y_test.min(), y_pred.min())
max_price = max(y_test.max(), y_pred.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--"
)

plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")

plt.show()


# 9. FEATURE IMPORTANCE


# Get the fitted preprocessing step
fitted_preprocessor = gradient_model.named_steps["preprocessor"]

# Get feature names after one-hot encoding
feature_names = fitted_preprocessor.get_feature_names_out()

# Get feature importance from Gradient Boosting
model = gradient_model.named_steps["model"]

importances = model.feature_importances_

# Create a DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Sort by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== TOP 20 FEATURES =====")
print(importance_df.head(20))


# 11. SAVE FINAL MODEL

import joblib

joblib.dump(gradient_model, "real_estate_model.pkl")

print("\nFinal model saved successfully!")
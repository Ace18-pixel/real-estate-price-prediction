import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv("train.csv")


# BASIC DATASET INFORMATION

print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

# SALE PRICE ANALYSIS

print("\nSale Price statistics:")
print(df["SalePrice"].describe())

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(20))

print("\nData types:")
print(df.dtypes.value_counts())


# SKEWNESS

print("\nTarget skewness:")

original_skewness = df["SalePrice"].skew()
log_price = np.log1p(df["SalePrice"])
log_skewness = log_price.skew()

print("Original skewness:", original_skewness)
print("Log-transformed skewness:", log_skewness)


# SALE PRICE DISTRIBUTION

plt.figure(figsize=(8, 5))

plt.hist(df["SalePrice"], bins=30)

plt.xlabel("Sale Price")
plt.ylabel("Number of Houses")
plt.title("Distribution of Sale Prices")

plt.show()


# LOG-TRANSFORMED DISTRIBUTION

plt.figure(figsize=(8, 5))

plt.hist(log_price, bins=30)

plt.xlabel("Log(Sale Price)")
plt.ylabel("Number of Houses")
plt.title("Distribution of Log-Transformed Sale Prices")

plt.show()

# NUMERICAL FEATURE CORRELATION

numeric_df = df.select_dtypes(include=["int64", "float64"])

correlations = numeric_df.corr()["SalePrice"].sort_values(ascending=False)

print("\nTop features correlated with SalePrice:")
print(correlations.head(15))

# RELATIONSHIP WITH SALE PRICE

plt.figure(figsize=(8, 5))

plt.scatter(df["GrLivArea"], df["SalePrice"], alpha=0.5)

plt.xlabel("Above Ground Living Area (sq ft)")
plt.ylabel("Sale Price")
plt.title("Living Area vs Sale Price")

plt.show()


plt.figure(figsize=(8, 5))

plt.scatter(df["OverallQual"], df["SalePrice"], alpha=0.5)

plt.xlabel("Overall Quality")
plt.ylabel("Sale Price")
plt.title("Overall Quality vs Sale Price")

plt.show()

# FEATURE ENGINEERING

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

print("\nNew features created:")
print(df[[
    "PropertyAge",
    "RemodAge",
    "TotalSF",
    "TotalBathrooms",
    "TotalPorchSF"
]].head())


# SEPARATE FEATURES AND TARGET

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# ID is just an identifier, not a useful predictive feature
X = X.drop("Id", axis=1)

print("\nFeatures:", X.shape)
print("Target:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# PREPROCESSING

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Identify numerical and categorical columns
numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns

print("\nNumber of numerical features:", len(numeric_features))
print("Number of categorical features:", len(categorical_features))


# Numerical preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])


# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Combine both
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nPreprocessing pipeline created successfully.")

# MODEL 1: LINEAR REGRESSION

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

linear_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Train
linear_model.fit(X_train, y_train)

# Predict
y_pred_linear = linear_model.predict(X_test)

# Evaluate
mae_linear = mean_absolute_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
r2_linear = r2_score(y_test, y_pred_linear)

print("\n===== LINEAR REGRESSION =====")
print("MAE :", mae_linear)
print("RMSE:", rmse_linear)
print("R²  :", r2_linear)

# MODEL 2: DECISION TREE

from sklearn.tree import DecisionTreeRegressor

tree_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(
        random_state=42,
        max_depth=10
    ))
])

# Train
tree_model.fit(X_train, y_train)

# Predict
y_pred_tree = tree_model.predict(X_test)

# Evaluate
mae_tree = mean_absolute_error(y_test, y_pred_tree)
rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))
r2_tree = r2_score(y_test, y_pred_tree)

print("\n===== DECISION TREE =====")
print("MAE :", mae_tree)
print("RMSE:", rmse_tree)
print("R²  :", r2_tree)


# MODEL 3: RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor

forest_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

# Train
forest_model.fit(X_train, y_train)

# Predict
y_pred_forest = forest_model.predict(X_test)

# Evaluate
mae_forest = mean_absolute_error(y_test, y_pred_forest)
rmse_forest = np.sqrt(mean_squared_error(y_test, y_pred_forest))
r2_forest = r2_score(y_test, y_pred_forest)

print("\n===== RANDOM FOREST =====")
print("MAE :", mae_forest)
print("RMSE:", rmse_forest)
print("R²  :", r2_forest)


# MODEL 4: GRADIENT BOOSTING

from sklearn.ensemble import GradientBoostingRegressor

gradient_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])

# Train
gradient_model.fit(X_train, y_train)

# Predict
y_pred_gradient = gradient_model.predict(X_test)

# Evaluate
mae_gradient = mean_absolute_error(y_test, y_pred_gradient)
rmse_gradient = np.sqrt(mean_squared_error(y_test, y_pred_gradient))
r2_gradient = r2_score(y_test, y_pred_gradient)

print("\n===== GRADIENT BOOSTING =====")
print("MAE :", mae_gradient)
print("RMSE:", rmse_gradient)
print("R²  :", r2_gradient)

# GRADIENT BOOSTING
# LOG-TRANSFORMED TARGET

# Transform target
y_train_log = np.log1p(y_train)

# Create model
gradient_log_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])

# Train using log-transformed target
gradient_log_model.fit(X_train, y_train_log)

# Predict log(price)
y_pred_log = gradient_log_model.predict(X_test)

# Convert predictions back to original price scale
y_pred_gradient_log = np.expm1(y_pred_log)

# Evaluate on original dollar scale
mae_gradient_log = mean_absolute_error(
    y_test,
    y_pred_gradient_log
)

rmse_gradient_log = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_gradient_log
    )
)

r2_gradient_log = r2_score(
    y_test,
    y_pred_gradient_log
)

print("\n===== GRADIENT BOOSTING + LOG TARGET =====")
print("MAE :", mae_gradient_log)
print("RMSE:", rmse_gradient_log)
print("R²  :", r2_gradient_log)
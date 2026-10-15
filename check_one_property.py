
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Load the data and the saved model
df = pd.read_csv("train.csv")
model = joblib.load("real_estate_model.pkl")

# Recreate the same 80/20 split
X = df.drop(columns=["SalePrice", "Id"])
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Use the same first test property we selected earlier
idx = X_test.index[0]
house = df.loc[[idx]].copy()

# Apply exactly the same feature engineering as training
house["PropertyAge"] = house["YrSold"] - house["YearBuilt"]
house["RemodAge"] = house["YrSold"] - house["YearRemodAdd"]

house["TotalSF"] = (
    house["TotalBsmtSF"].fillna(0)
    + house["1stFlrSF"].fillna(0)
    + house["2ndFlrSF"].fillna(0)
)

house["TotalBathrooms"] = (
    house["FullBath"].fillna(0)
    + 0.5 * house["HalfBath"].fillna(0)
)

house["TotalPorchSF"] = (
    house["OpenPorchSF"].fillna(0)
    + house["3SsnPorch"].fillna(0)
    + house["EnclosedPorch"].fillna(0)
    + house["ScreenPorch"].fillna(0)
    + house["WoodDeckSF"].fillna(0)
)

# Predict with the property's complete original feature row
X_one = house.drop(columns=["SalePrice", "Id"])
prediction = model.predict(X_one)[0]
actual = house["SalePrice"].iloc[0]

print("Property ID:", house["Id"].iloc[0])
print(f"Actual price:          ${actual:,.2f}")
print(f"Full-row prediction:   ${prediction:,.2f}")
print(f"Absolute error:        ${abs(actual - prediction):,.2f}")
print(f"Percentage error:      {abs(actual - prediction) / actual * 100:.2f}%")

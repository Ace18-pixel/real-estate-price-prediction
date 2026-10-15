
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")

# Reproduce the same split used during training
X = df.drop(columns=["SalePrice", "Id"])
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Select one property from the held-out test set
idx = X_test.index[0]
house = df.loc[idx]

fields = {
    "Overall Quality": "OverallQual",
    "Living Area": "GrLivArea",
    "Year Built": "YearBuilt",
    "Year Remodeled": "YearRemodAdd",
    "Basement Area": "TotalBsmtSF",
    "First Floor Area": "1stFlrSF",
    "Second Floor Area": "2ndFlrSF",
    "Lot Area": "LotArea",
    "Garage Cars": "GarageCars",
    "Full Bathrooms": "FullBath",
    "Half Bathrooms": "HalfBath",
    "Neighborhood": "Neighborhood",
}

print("TEST PROPERTY — ID:", house["Id"])
print("\nEnter these values in your website:\n")

for label, column in fields.items():
    print(f"{label}: {house[column]}")

print(f"\nACTUAL SALE PRICE: ${house['SalePrice']:,.2f}")

import pandas as pd
import joblib


# 1. LOAD TRAINED MODEL

model = joblib.load("real_estate_model.pkl")

# Load original training data
df = pd.read_csv("train.csv")


# 2. CREATE A BASE PROPERTY

property_data = {}

for column in df.columns:

    if column == "SalePrice":
        continue

    # Numeric columns → use median
    if pd.api.types.is_numeric_dtype(df[column]):
        property_data[column] = df[column].median()

    # Categorical/text columns → use most common value
    else:
        property_data[column] = df[column].mode()[0]


property_df = pd.DataFrame([property_data])


# 3. USER INPUTS


print("\n===================================")
print("     REAL ESTATE VALUATION ENGINE")
print("===================================\n")

overall_qual = int(input("Overall Quality (1-10): "))
gr_liv_area = float(input("Living Area (sq ft): "))
year_built = int(input("Year Built: "))
year_remod = int(input("Year Remodeled: "))

total_bsmt_sf = float(input("Basement Area (sq ft): "))
first_flr_sf = float(input("1st Floor Area (sq ft): "))
second_flr_sf = float(input("2nd Floor Area (sq ft): "))

lot_area = float(input("Lot Area (sq ft): "))
garage_cars = float(input("Garage Capacity (cars): "))

full_bath = float(input("Full Bathrooms: "))
half_bath = float(input("Half Bathrooms: "))

neighborhood = input("Neighborhood (e.g. NAmes, CollgCr, NridgHt): ")


# 4. UPDATE PROPERTY INFORMATION

property_df["OverallQual"] = overall_qual
property_df["GrLivArea"] = gr_liv_area
property_df["YearBuilt"] = year_built
property_df["YearRemodAdd"] = year_remod

property_df["TotalBsmtSF"] = total_bsmt_sf
property_df["1stFlrSF"] = first_flr_sf
property_df["2ndFlrSF"] = second_flr_sf

property_df["LotArea"] = lot_area
property_df["GarageCars"] = garage_cars

property_df["FullBath"] = full_bath
property_df["HalfBath"] = half_bath

property_df["Neighborhood"] = neighborhood


# 5. FEATURE ENGINEERING

# Ames dataset records the year of sale
# Use the property's sale year from the
# base dataset.

property_df["PropertyAge"] = (
    property_df["YrSold"] - property_df["YearBuilt"]
)

property_df["RemodAge"] = (
    property_df["YrSold"] - property_df["YearRemodAdd"]
)

property_df["TotalSF"] = (
    property_df["TotalBsmtSF"]
    + property_df["1stFlrSF"]
    + property_df["2ndFlrSF"]
)

property_df["TotalBathrooms"] = (
    property_df["FullBath"]
    + 0.5 * property_df["HalfBath"]
    + property_df["BsmtFullBath"]
    + 0.5 * property_df["BsmtHalfBath"]
)

property_df["TotalPorchSF"] = (
    property_df["OpenPorchSF"]
    + property_df["EnclosedPorch"]
    + property_df["3SsnPorch"]
    + property_df["ScreenPorch"]
)


# 6. REMOVE ID

property_df = property_df.drop(columns=["Id"])


# 7. PREDICT PRICE

predicted_price = model.predict(property_df)[0]


# 8. DISPLAY RESULT

print("\n===================================")
print("        PROPERTY VALUATION")
print("===================================")

print(f"\nEstimated House Price: ${predicted_price:,.2f}")

print("\n===================================")
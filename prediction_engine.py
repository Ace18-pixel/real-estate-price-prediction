
import pandas as pd
import joblib


# LOAD TRAINED MODEL AND DATASET


model = joblib.load("real_estate_model.pkl")
df = pd.read_csv("train.csv")


# PREDICTION FUNCTION


def predict_house_price(
    overall_qual,
    gr_liv_area,
    year_built,
    year_remod,
    total_bsmt_sf,
    first_flr_sf,
    second_flr_sf,
    lot_area,
    garage_cars,
    full_bath,
    half_bath,
    neighborhood
):

    
    # Create a base property using medians
    # and modes for unspecified features
    

    property_data = {}

    for column in df.columns:

        if column in ["SalePrice", "Id"]:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            property_data[column] = df[column].median()
        else:
            property_data[column] = df[column].mode()[0]

    property_df = pd.DataFrame([property_data])


   
    # Apply user-provided values


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



    # Feature engineering
    # Must match final_model.py
    

    property_df["PropertyAge"] = (
        property_df["YrSold"] - property_df["YearBuilt"]
    )

    property_df["RemodAge"] = (
        property_df["YrSold"] - property_df["YearRemodAdd"]
    )

    property_df["TotalSF"] = (
        property_df["TotalBsmtSF"].fillna(0)
        + property_df["1stFlrSF"].fillna(0)
        + property_df["2ndFlrSF"].fillna(0)
    )

    property_df["TotalBathrooms"] = (
        property_df["FullBath"].fillna(0)
        + 0.5 * property_df["HalfBath"].fillna(0)
    )

    property_df["TotalPorchSF"] = (
        property_df["OpenPorchSF"].fillna(0)
        + property_df["3SsnPorch"].fillna(0)
        + property_df["EnclosedPorch"].fillna(0)
        + property_df["ScreenPorch"].fillna(0)
        + property_df["WoodDeckSF"].fillna(0)
    )



    # Predict using the saved ML pipeline
   

    predicted_price = model.predict(property_df)[0]

    return float(predicted_price)



# TEST THE PREDICTION ENGINE DIRECTLY

if __name__ == "__main__":

    price = predict_house_price(
        overall_qual=8,
        gr_liv_area=2000,
        year_built=2015,
        year_remod=2015,
        total_bsmt_sf=1000,
        first_flr_sf=1200,
        second_flr_sf=800,
        lot_area=9000,
        garage_cars=2,
        full_bath=2,
        half_bath=1,
        neighborhood="NAmes"
    )

    print(f"\nPredicted House Price: ${price:,.2f}")

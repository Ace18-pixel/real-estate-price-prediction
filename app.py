from flask import Flask, render_template, request
from prediction_engine import predict_house_price

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        quality = int(request.form["quality"])
        living_area = float(request.form["living_area"])
        year_built = int(request.form["year_built"])
        year_remodeled = int(request.form["year_remodeled"])
        basement_area = float(request.form["basement_area"])
        first_floor_area = float(request.form["first_floor_area"])
        second_floor_area = float(request.form["second_floor_area"])
        lot_area = float(request.form["lot_area"])
        garage_cars = int(request.form["garage_cars"])
        full_bath = int(request.form["full_bath"])
        half_bath = int(request.form["half_bath"])
        neighborhood = request.form["neighborhood"]

        prediction = predict_house_price(
            overall_qual=quality,
            gr_liv_area=living_area,
            year_built=year_built,
            year_remod=year_remodeled,
            total_bsmt_sf=basement_area,
            first_flr_sf=first_floor_area,
            second_flr_sf=second_floor_area,
            lot_area=lot_area,
            garage_cars=garage_cars,
            full_bath=full_bath,
            half_bath=half_bath,
            neighborhood=neighborhood
)
        

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
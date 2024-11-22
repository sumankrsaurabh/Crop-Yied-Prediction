import json
import pickle
import warnings
import pandas as pd
from flask import Flask, request, jsonify, render_template

warnings.filterwarnings("ignore")

# Load the saved model and encoder
with open("crop_yield_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("ordinal_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load the country-item mapping JSON file
with open("country_item_dict.json", "r") as f:
    country_item_dict_local = json.load(f)

# Initialize the Flask app
app = Flask(__name__)


# Home route (for the form)
@app.route("/")
def home():
    countries = list(country_item_dict_local.keys())
    # print(countries)
    return render_template("index.html", countries=countries)


# Prediction route (for predicting crop yield)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validate input data
        required_fields = [
            "Area",
            "Item",
            "Year",
            "average_rain_fall_mm_per_year",
            "pesticides_tonnes",
            "avg_temp",
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Extract input values
        area = data["Area"]
        item = data["Item"]
        year = int(data["Year"])
        rainfall = float(data["average_rain_fall_mm_per_year"])
        pesticides = float(data["pesticides_tonnes"])
        temp = float(data["avg_temp"])

        # Prepare the input for prediction
        sample_input = {
            "Area": [area],
            "Item": [item],
            "Year": [year],
            "average_rain_fall_mm_per_year": [rainfall],
            "pesticides_tonnes": [pesticides],
            "avg_temp": [temp],
        }

        # Convert categorical columns using the encoder
        sample_df = pd.DataFrame(sample_input)
        sample_df[["Area", "Item"]] = encoder.transform(sample_df[["Area", "Item"]])

        # Make prediction
        prediction = model.predict(sample_df)[0]

        # Return the prediction as a JSON response
        return jsonify({"predicted_yield": round(prediction, 2)})

    except ValueError as e:
        return jsonify({"error": f"Invalid input value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/get_items", methods=["POST"])
def get_items():
    try:
        data = request.get_json()
        area = data.get("Area")

        if not area:
            return jsonify({"error": "Area is required"}), 400

        # Fetch items for the given area
        items = country_item_dict_local.get(area, [])
        # print(items)
        return jsonify({"items": items})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# Run the app
if __name__ == "__main__":
    app.run(debug=True)

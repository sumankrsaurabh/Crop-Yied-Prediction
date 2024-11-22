import warnings
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import pickle
import json

warnings.filterwarnings("ignore")

# Load and preprocess the dataset
df = pd.read_csv("yield_df.csv")
df.drop("Unnamed: 0", axis=1, inplace=True, errors="ignore")

# Filter countries with less than 100 data points
countries_to_drop = df["Area"].value_counts()[lambda x: x < 100].index
df = df[~df["Area"].isin(countries_to_drop)].reset_index(drop=True)

# Extract unique countries and corresponding items
country_item_dict = df.groupby("Area")["Item"].unique().to_dict()

# Ensure all items are strings or serializable
# Convert items to strings if necessary
for country in country_item_dict:
    country_item_dict[country] = [str(item) for item in country_item_dict[country]]

# Save to a JSON file
with open("country_item_dict.json", "w") as json_file:
    json.dump(country_item_dict, json_file, indent=4)

print("Country-Item list saved to country_item_dict.json")

# Encode categorical variables
categorical_columns = ["Area", "Item"]
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df[categorical_columns] = encoder.fit_transform(df[categorical_columns])

# Split data into features and target
X = df[
    [
        "Area",
        "Item",
        "Year",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
    ]
]
y = df["hg/ha_yield"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save the model to a file using pickle
model_file = "crop_yield_model.pkl"
with open(model_file, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to {model_file}")

# Save the encoder to a file using pickle
encoder_file = "ordinal_encoder.pkl"
with open(encoder_file, "wb") as f:
    pickle.dump(encoder, f)
print(f"Encoder saved to {encoder_file}")

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Train Accuracy: {model.score(X_train, y_train) * 100:.2f}%")
print(f"Test Accuracy: {model.score(X_test, y_test) * 100:.2f}%")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")


# Predict for a sample input
sample_input = pd.DataFrame(
    {
        "Area": ["India"],
        "Item": ["Wheat"],
        "Year": [2024],
        "average_rain_fall_mm_per_year": [1200],
        "pesticides_tonnes": [450],
        "avg_temp": [25],
    }
)
sample_input[categorical_columns] = encoder.transform(sample_input[categorical_columns])
predicted_yield = model.predict(sample_input)
print(f"\nPredicted Yield (hg/ha) for Sample Input: {predicted_yield[0]:.2f}")

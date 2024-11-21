# Crop Yield Prediction

A web-based application for predicting crop output based on environmental and agricultural factors. This project helps farmers, researchers, and policymakers make informed decisions by providing accurate predictions of crop output per unit area.

## Features
- **Dynamic Dropdowns**: Items available for selection update based on the chosen area.
- **Accurate Predictions**: Leveraging machine learning models for crop output prediction.
- **User-Friendly Interface**: A simple and intuitive UI for inputting data and viewing results.
- **Flexible Inputs**: Supports data like rainfall, pesticides used, temperature, area, and crop type.

## Technologies Used
- **Frontend**: HTML, CSS (custom styles), JavaScript
- **Backend**: Python (Flask framework)
- **Machine Learning**: Scikit-learn (for training and prediction models)
- **Data Handling**: Pandas for data processing, Pickle for model serialization

## Prerequisites
Before running the project, ensure you have the following installed:
- Python 3.7 or higher
- Flask
- Required Python libraries: `pandas`, `scikit-learn`

## Project Structure
```plaintext
├── static/
│   ├── images/
│   │   └── background.jpg
├── templates/
│   └── index.html
├── crop_yield_model.pkl          # Trained machine learning model
├── ordinal_encoder.pkl           # Encoder for categorical variables
├── country_item_dict.json        # JSON mapping countries to items
├── app.py                        # Flask app script
└── README.md                     # This file

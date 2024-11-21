import warnings

warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv("yield_df.csv")
df.drop("Unnamed: 0", axis=1, inplace=True)


country_counts = df["Area"].value_counts()
countries_to_drop = country_counts[country_counts < 100].index.tolist()
df_filtered = df[~df["Area"].isin(countries_to_drop)]
df = df_filtered.reset_index(drop=True)

datacorr = df.copy()


categorical_columns = datacorr.select_dtypes(include=["object"]).columns.tolist()
label_encoder = LabelEncoder()
for column in categorical_columns:
    datacorr[column] = label_encoder.fit_transform(datacorr[column])


X, y = datacorr.drop(labels="hg/ha_yield", axis=1), datacorr["hg/ha_yield"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

results = []

models = [("Random Forest", RandomForestRegressor(random_state=42))]

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    MSE = mean_squared_error(y_test, y_pred)
    R2_score = r2_score(y_test, y_pred)
    results.append((name, accuracy, MSE, R2_score))
    acc = model.score(X_train, y_train) * 100
    print(f"The accuracy of the {name} Model Train is {acc:.2f}")
    acc = model.score(X_test, y_test) * 100
    print(f"The accuracy of the  {name} Model Test is {acc:.2f}")

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt 
# import seaborn as sns
# import numpy as np
# import geopandas as gpd
# import plotly.express as px

df = pd.read_csv("yield_df.csv")
df.drop("Unnamed: 0", axis=1, inplace=True)


country_counts = df["Area"].value_counts()
countries_to_drop = country_counts[country_counts < 100].index.tolist()
df_filtered = df[~df["Area"].isin(countries_to_drop)]
df = df_filtered.reset_index(drop=True)

datacorr = df.copy()

from sklearn.preprocessing import LabelEncoder

categorical_columns = datacorr.select_dtypes(include=['object']).columns.tolist()
label_encoder = LabelEncoder()
for column in categorical_columns:
    datacorr[column] = label_encoder.fit_transform(datacorr[column])

# sns.heatmap(datacorr.corr() ,annot=True , cmap='PuOr')
# sns.set_theme(palette='BrBG')
# df.hist(figsize=(5,10))
# sns.pairplot(data=df,hue='Item',kind='scatter',palette='BrBG')
# df2=df[df['Item']=='Yams']
# df2.groupby('Year')['hg/ha_yield'].mean().plot(color='brown')
# geojson_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
# data = gpd.read_file(geojson_url)

# merged_data = data.merge(df, left_on='NAME', right_on='Area', how='left')
# fig, ax = plt.subplots(figsize=(15, 10))
# merged_data.plot( column='hg/ha_yield', cmap='Greens_r', linewidth=0.8, edgecolor='0.8')
# plt.title("Countries")
# # plt.show()

# del merged_data
# del data

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
# from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import BaggingRegressor

X, y = datacorr.drop(labels='hg/ha_yield', axis=1), datacorr['hg/ha_yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

results = []

models = [
    # ('Linear Regression', LinearRegression()),
    ('Random Forest', RandomForestRegressor(random_state=42)),
    # ('Gradient Boost', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,random_state=42)),
    # ('XGBoost', XGBRegressor(random_state=42)),
    # ('KNN',KNeighborsRegressor(n_neighbors=5)),
    # ('Decision Tree',DecisionTreeRegressor(random_state=42)),
    # ('Bagging Regressor',BaggingRegressor(n_estimators=150, random_state=42))
          ]

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    MSE = mean_squared_error(y_test, y_pred)
    R2_score = r2_score(y_test, y_pred)
    results.append((name, accuracy, MSE, R2_score))
    acc = (model.score(X_train , y_train)*100)
    print(f'The accuracy of the {name} Model Train is {acc:.2f}')
    acc =(model.score(X_test , y_test)*100)
    print(f'The accuracy of the  {name} Model Test is {acc:.2f}')    
#     plt.scatter(y_test, y_pred,s=10,color='#9B673C')
#     plt.xlabel('Actual Values')
#     plt.ylabel('Predicted Values')
# #     plt.title(f' {name} Evaluation')
#     plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='green', linewidth = 4)
#     plt.show()

dff = pd.DataFrame(results, columns=['Model', 'Accuracy', 'MSE', 'R2_score'])
df_styled_best = dff.style.highlight_max(subset=['Accuracy','R2_score'], color='green').highlight_min(subset=['MSE'], color='green').highlight_max(subset=['MSE'], color='red').highlight_min(subset=['Accuracy','R2_score'], color='red')

# df_styled_worst = dff.style.highlight_max(subset=['MSE'], color='red').highlight_min(subset=['Accuracy','R2_score'], color='red')

print(df_styled_best)
# display(df_styled_worst)
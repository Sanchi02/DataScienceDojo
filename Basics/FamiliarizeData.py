# The steps to building and using a model are:

# Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
# Fit: Capture patterns from provided data. This is the heart of modeling.
# Predict: Just what it sounds like
# Evaluate: Determine how accurate the model's predictions are.

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

filePath = 'C:\\Users\\Sanchi\\Desktop\\Stuff\\DataScienceSiraj\\data\\melb_data.csv'
melbourne_data = pd.read_csv(filePath)

# melbourne_data.describe() 
# print(melbourne_data.columns)
melbourne_data = melbourne_data.dropna(axis=0)
y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
# print(X.describe())
# print(X.head())
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
melbourne_model = DecisionTreeRegressor(random_state=1)
# melbourne_model.fit(X, y)
melbourne_model.fit(train_X, train_y)
# print(melbourne_model.predict(X.head()))
# predicted_home_prices = melbourne_model.predict(X)
# print(mean_absolute_error(y, predicted_home_prices))
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
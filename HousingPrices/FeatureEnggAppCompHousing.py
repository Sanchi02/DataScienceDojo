import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import KFold
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

train_file_path = 'C:\\Users\\Sanchi\\Desktop\\Stuff\\DataScienceSiraj\\data\\housing_train.csv'
train_data = pd.read_csv(train_file_path)

test_data_path = 'C:\\Users\\Sanchi\\Desktop\\Stuff\\DataScienceSiraj\\data\\housing_test.csv'
test_data = pd.read_csv(test_data_path)

# Since, I decided to go with LotFrontage as added parameter,I first need to predict LotFrontage, 
# as some of the rows have NA as value. 

# Trying to avoid data leakage
train_data = train_data.drop('SalePrice', axis=1)
all_data = pd.concat([train_data, test_data], ignore_index=True).copy()

# All the data which has values for LotFrontage is separated from the ones that have NA has value. 
# Thus, we get training and testing sets for LotFrontage
test  = all_data[all_data.LotFrontage.isnull()]
train = all_data[~all_data.LotFrontage.isnull()]
target = train.LotFrontage

# After plotting various relationships and shooting a few arrows in darkness,
# I decided that these would be the parameters affecting the LotFrontageArea.
features_for_lotFrontage = ['LotArea', 'LotConfig', 'LotShape', 'Alley', 'MSZoning', 'BldgType', 'Neighborhood', 'Condition1', 'Condition2', 'GarageCars']
X_train_lotFrontage = train[features_for_lotFrontage]
y_lotFrontage = train['LotFrontage']

# Since a lot of the chosen parameters have non-numeric values, I use get_dummies function for converting
# them into numeric values inorder to simplify our training. Once categorical variables are converted,
# we normalize the data
X_train_lotFrontage = pd.get_dummies(X_train_lotFrontage)
X_train_lotFrontage = (X_train_lotFrontage - X_train_lotFrontage.mean())/X_train_lotFrontage.std()
X_train_lotFrontage = X_train_lotFrontage.fillna(0)

# After testing a few models and tuning their parameters, SVM is the model I choose for predicting the 
# LotFrontage.
clf = svm.SVR(kernel='rbf', C=100, gamma=0.001)
inital_score = 0

# Using the K-fold method to train the model.
kf = KFold(n_splits=10, shuffle=True, random_state=3)
for trn, tst in kf.split(train):
    clf.fit(X_train_lotFrontage.iloc[trn], y_lotFrontage.iloc[trn])

# Since the testing data also has a not of NA values for Lotfrontage we need to predict them using the 
# same method we predicted them in case of training data
X_test = test[features_for_lotFrontage]
X_test = pd.get_dummies(X_test)
X_test = (X_test - X_test.mean())/X_test.std()
X_test = X_test.fillna(0)

# Make sure that dummy columns from training set are replicated in test set
for col in (set(X_train_lotFrontage.columns) - set(X_test.columns)):
    X_test[col] = 0
X_test = X_test[X_train_lotFrontage.columns]

# Adding the predicted data to the combined dataset of training and testing data
all_data.loc[all_data.LotFrontage.isnull(), 'LotFrontage'] = clf.predict(X_test)

# Once we have the predicted values for our data, we segregate the training and testing data as it was before,
# with just a teeny difference of added LotFrontage values
X_train_post = all_data[:train_data.shape[0]]
X_test_post = all_data[train_data.shape[0]:]

# Setting the X and Y for training the model
features = ['LotArea', 'LotFrontage', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X_train_post = X_train_post[features]
y = train_data.SalePrice

# After trying out various models and parameter tuning,
rf_model_on_full_data = RandomForestRegressor(random_state=1, n_estimators=70)
rf_model_on_full_data.fit(X_train_post,y)

test_X = X_test_post[features]
test_preds = rf_model_on_full_data.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id,
                      'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)


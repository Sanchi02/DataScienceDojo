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
test_data_path = 'C:\\Users\\Sanchi\\Desktop\\Stuff\\DataScienceSiraj\\data\\housing_test.csv'

train_data_raw = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_data_path)

features = ['LotArea', 'LotFrontage', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = train_data_raw[features]
y = train_data_raw.SalePrice

train_data = train_data_raw.drop('SalePrice', axis=1)
all_data = pd.concat([train_data, test_data], ignore_index=True).copy()

test  = all_data[all_data.LotFrontage.isnull()]
train = all_data[~all_data.LotFrontage.isnull()]
target = train.LotFrontage
features_for_lotFrontage = ['LotArea', 'LotConfig', 'LotShape', 'Alley', 'MSZoning', 'BldgType', 'Neighborhood', 'Condition1', 'Condition2', 'GarageCars']
X_train_lotFrontage = train[features_for_lotFrontage]
y_lotFrontage = train['LotFrontage']

X_train_lotFrontage = pd.get_dummies(X_train_lotFrontage)
X_train_lotFrontage = (X_train_lotFrontage - X_train_lotFrontage.mean())/X_train_lotFrontage.std()
X_train_lotFrontage = X_train_lotFrontage.fillna(0)

clf = svm.SVR(kernel='rbf', C=100, gamma=0.001)
inital_score = 0
kf = KFold(n_splits=10, shuffle=True, random_state=3)


for trn, tst in kf.split(train):
    clf.fit(X_train_lotFrontage.iloc[trn], y_lotFrontage.iloc[trn])

X_test = test[features_for_lotFrontage]
X_test = pd.get_dummies(X_test)
X_test = (X_test - X_test.mean())/X_test.std()
X_test = X_test.fillna(0)


for col in (set(X_train_lotFrontage.columns) - set(X_test.columns)):
    X_test[col] = 0

X_test = X_test[X_train_lotFrontage.columns]
all_data.loc[all_data.LotFrontage.isnull(), 'LotFrontage'] = clf.predict(X_test)

X_train_post = all_data[:train_data.shape[0]]
X_test_post = all_data[train_data.shape[0]:]

X_train_post = X_train_post[features]

rf_model_on_full_data = RandomForestRegressor(random_state=1, n_estimators=70)
rf_model_on_full_data.fit(X_train_post,y)

test_X = X_test_post[features]

test_preds = rf_model_on_full_data.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id,
                      'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)


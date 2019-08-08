import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import warnings
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold

warnings.simplefilter(action='ignore', category=FutureWarning)

def idOutliers(dat):
    tile25 = dat.describe()[4]
    tile75 = dat.describe()[6]
    iqr = tile75 - tile25 
    out = (dat > tile75+1.5*iqr) | (dat < tile25-1.5*iqr)
    return out

train_file_path = 'C:\\Users\\Sanchi\\Desktop\\Stuff\\DataScienceSiraj\\data\\housing_train.csv'
test_data_path = 'C:\\Users\\Sanchi\\Desktop\\Stuff\\DataScienceSiraj\\data\\housing_test.csv'

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_data_path)

y = train_data.SalePrice
features = ['LotArea', 'LotFrontage', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = train_data[features]

#Trying to prevent "Data leakage"
train_data = train_data.drop('SalePrice', axis=1)
all_data = pd.concat([train_data, test_data], ignore_index=True).copy()

test  = all_data[all_data.LotFrontage.isnull()]
train = all_data[~all_data.LotFrontage.isnull()]
target = train.LotFrontage

y_lotFrontage = train['LotFrontage']
X_train_lotFrontage = train.loc[:,['LotArea', 'LotConfig', 'LotShape', 'Alley', 'MSZoning', 'BldgType', 'Neighborhood', 'Condition1', 'Condition2', 'GarageCars']]

X_train_lotFrontage = pd.get_dummies(X_train_lotFrontage)
X_train_lotFrontage = (X_train_lotFrontage - X_train_lotFrontage.mean())/X_train_lotFrontage.std()
X_train_lotFrontage = X_train_lotFrontage.fillna(0)

clf = svm.SVR(kernel='rbf', C=100, gamma=0.001)
inital_score = 0
kf = KFold(n_splits=10, shuffle=True, random_state=3) 



for trn, tst in kf.split(train):
    # Compute benchmark score prediction based on mean neighbourhood LotFrontage
    fold_train_samples = train.iloc[trn]
    fold_test_samples = train.iloc[tst]
    neigh_means = fold_train_samples.groupby('Neighborhood')['LotFrontage'].mean()
    all_mean = fold_train_samples['LotFrontage'].mean()
    y_pred_neigh_means = fold_test_samples.join(neigh_means, on = 'Neighborhood', lsuffix='benchmark')['LotFrontage']
    y_pred_all_mean = [all_mean] * fold_test_samples.shape[0]
    
    # Compute benchmark score prediction based on overall mean LotFrontage
    u1 = ((fold_test_samples['LotFrontage'] - y_pred_neigh_means) ** 2).sum()
    u2 = ((fold_test_samples['LotFrontage'] - y_pred_all_mean) ** 2).sum()
    v = ((fold_test_samples['LotFrontage'] - fold_test_samples['LotFrontage'].mean()) ** 2).sum()
    
    # Perform model fitting 
    clf.fit(X_train_lotFrontage.iloc[trn], y_lotFrontage.iloc[trn])
    
    # Record all scores for averaging
    inital_score = inital_score + mean_absolute_error(fold_test_samples['LotFrontage'], clf.predict(X_train_lotFrontage.iloc[tst]))
    
X_test = test.loc[:,['LotArea', 'LotConfig', 'LotShape', 'Alley', 'MSZoning', 'BldgType', 'Neighborhood', 'Condition1', 'Condition2', 'GarageCars']]
X_test = pd.get_dummies(X_test)
X_test = (X_test - X_test.mean())/X_test.std()
X_test = X_test.fillna(0)


for col in (set(X_train_lotFrontage.columns) - set(X_test.columns)):
    X_test[col] = 0

X_test = X_test[X_train_lotFrontage.columns]
all_data.loc[all_data.LotFrontage.isnull(), 'LotFrontage'] = clf.predict(X_test)



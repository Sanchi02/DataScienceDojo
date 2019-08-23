import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
# Ignore FutureWarnings related to internal pandas and np code
import warnings


iowa_file_path = 'C:\\Users\\Sanchi\\Desktop\\Stuff\\DataScienceSiraj\\data\\housing_train.csv'
raw_train = pd.read_csv(iowa_file_path)
features = ['LotArea', 'LotFrontage', 'YearBuilt', 'SalePrice', 'TotRmsAbvGrd']
raw_train = raw_train[features]
sns.pairplot(raw_train)
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 12:13:16 2020

@author: HI
"""
#load the dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
pd.pandas.set_option('display.max_columns',None)
dataset  = pd.read_csv("train.csv")
#printig the missing values
pd.pandas.set_option('display.max_rows',None)
print(dataset.isnull().sum())
print(dataset.info())
#dealing with missing values
missing_values = [features for features in dataset.columns if dataset[features].isnull().sum() and dataset[features].dtypes == 'O']
for feature in missing_values:
    print(feature, np.round(dataset[feature].isnull().mean(), 4))
#Categorical features
def cat_features(data, missing_values):
    data = dataset.copy()
    data[missing_values] = data[missing_values].fillna(0)
    return data
dataset = cat_features(dataset, missing_values)
print(dataset[missing_values].isnull().sum())
#dealing  with numerical dat
num_values = [feature for feature in dataset.columns if dataset[feature].isnull().sum() and dataset[feature].dtypes != 'O' ]
print(num_values)
# dealing with temp
yv = [feature for feature in dataset.columns if 'Yr' in feature or 'Year' in feature]
print(yv)
# sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')
# dealing with numeric missing values
def num_variables(data, missing_values):
    data = dataset.copy()
    data['LotFrontage'] = dataset['LotFrontage'].fillna(dataset['LotFrontage'].mean())
    data['MasVnrArea']  = dataset['MasVnrArea'].fillna(dataset['MasVnrArea'].mean())
    data['GarageYrBlt']  = dataset['GarageYrBlt'].fillna(dataset['GarageYrBlt'].mean())
    return data
dataset = num_variables(dataset, missing_values)
print(dataset[num_values].isnull().sum())
# # dropping the categorical feature  with highest percentahe
dataset.drop(['PoolQC','Fence'], axis=1, inplace=True)
dataset.drop('MiscFeature', axis = 1, inplace=True)
dataset.drop('Alley' , axis = 1, inplace=True)
print(dataset.isnull().sum())
#converting object variables in numeric
# testing the  categorical
print(dataset['MSZoning'].value_counts())
print(dataset['Street'].value_counts())
print(dataset['GarageQual'].value_counts())
#from ahbove sample testing we can say that catedorical varibales should be converting intoone hot encoding
#onehot encoding
columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']
print(len(columns))
#Select categorical featur
#concating the dataset and testdataset
# sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')
test_dataset  = pd.read_csv("ModifyTesthrp.csv")
dt = pd.concat([dataset,test_dataset], axis=1)
dt = dt.loc[:,~dt.columns.duplicated()]
dt = pd.get_dummies(dt)
print(dt.shape)
print(dt.isnull().sum())
print(dt.describe())
#creating the target and independent varianle
y = dt['SalePrice']
dt.drop('SalePrice',axis = 1, inplace=  True)
X= dt               
#training the test the model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, random_state = 0 ,test_size =0.33)
X_test.fillna(X_test.mean())
print(X_test)
print(np.isnan(X_test))
print(np.where(np.isnan(X_test)))
print(X_test.fillna(X_test.mean()))
# #fitting the model using Xg_Boosting
import xgboost
from  xgboost import XGBRegressor
classifier=xgboost.XGBRegressor()
regressor=xgboost.XGBRegressor()
#doing HyperParameter Optimization
n_estimators = [100,300,600,900,1100,1500]
max_depth = [2,3,5,10,15]
booster = ['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]
#defining the grid of the parameters
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    }
# Set up the random search with 4-fold cross validation
from sklearn.model_selection import RandomizedSearchCV
randomsearch =  RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)
randomsearch.fit(X_train,y_train)
print(randomsearch.best_estimator_)
xgb =XGBRegressor(booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.6, gamma=0,
              importance_type='gain', learning_rate=0.01, max_delta_step=0,
              max_depth=4, min_child_weight=1.5, n_estimators=1500,
              n_jobs=1, nthread=None, objective='reg:linear',
              reg_alpha=0.6, reg_lambda=0.6, scale_pos_weight=1, 
              silent=None, subsample=0.8, verbosity=1)

print(xgb.fit(X_train,y_train))
y_pred =  xgb.predict(X_test)
from sklearn.metrics import mean_squared_log_error 
import math
print('RMSLE for XG Boost')
print(np.sqrt(mean_squared_log_error(y_test, y_pred) ))
#submission  of the final report
submission = pd.DataFrame({
        "Id": X_test["Id"],
        "SalePrice": y_pred
    })
submission.to_csv("submission.csv")


# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# Importing the Keras libraries and packages
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import ReLU, sigmoid
# from keras.layers import Dropout

# # Initialising the ANN
# classifier = Sequential()
# # Adding the input layer and the first hidden layer
# classifier.add(Dense(output_dim = 75, init = 'he_uniform', activation='relu',input_dim = 200))
# classifier.add(Dense(output_dim = 50, init = 'he_uniform',activation='relu'))
# classifier.add(Dense(output_dim = 25, init = 'he_uniform',activation='relu'))
# classifier.add(Dense(output_dim = 15, init = 'he_uniform',activation='relu'))
# classifier.add(Dense(output_dim = 5, init = 'he_uniform',activation='relu'))
# # Adding the OUTPUT Layer layer
# classifier.add(Dense(output_dim = 1, init = 'glorot_uniform', activation = 'sigmoid'))
# # Compiling the ANN
# classifier.compile(loss= root_mean_squared_error, optimizer='Adamax')

# # Fitting the ANN to the Training set
# model_history=classifier.fit(X_train.values, y_train.values,validation_split=0.20, batch_size = 10, nb_epoch = 1000)




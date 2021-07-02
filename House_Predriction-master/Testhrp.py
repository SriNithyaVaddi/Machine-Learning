# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 16:05:34 2020

@author: HI
"""
#load the dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
pd.pandas.set_option('display.max_columns',None)
test_dataset  = pd.read_csv("train.csv")
#printig the missing values
pd.pandas.set_option('display.max_rows',None)
print(test_dataset.isnull().sum())
print(test_dataset.info())

#dealing with missing values
pd.pandas.set_option('display.max_rows',None)
print(test_dataset.isnull().sum())
print(test_dataset.info())
#dealing with missing values
missing_values = [features for features in test_dataset.columns if test_dataset[features].isnull().sum() and test_dataset[features].dtypes == 'O']
for feature in missing_values:
    print(feature, np.round(test_dataset[feature].isnull().mean(), 4))
#Categorical features
def cat_features(data, missing_values):
    data = test_dataset.copy()
    data[missing_values] = data[missing_values].fillna(0)
    return data
test_dataset = cat_features(test_dataset, missing_values)
print(test_dataset[missing_values].isnull().sum())
test_dataset.drop(['PoolQC','Fence','MiscFeature'], axis=1, inplace=True)
test_dataset.drop('Alley', axis = 1, inplace=True)
test_dataset['LotFrontage'] = test_dataset['LotFrontage'].fillna(test_dataset['LotFrontage'].mean())
test_dataset['GarageType'] = test_dataset['GarageType'].fillna(test_dataset['GarageType'].mode()[0])
test_dataset['GarageYrBlt'] = test_dataset['GarageYrBlt'].fillna(test_dataset['GarageYrBlt'].mode()[0])
test_dataset['MasVnrType'] = test_dataset['MasVnrType'].fillna(test_dataset['MasVnrType'].mode()[0])
test_dataset['ExterCond'] = test_dataset['ExterCond'].fillna(test_dataset['ExterCond'].mode()[0])
test_dataset['Utilities']=test_dataset['Utilities'].fillna(test_dataset['Utilities'].mode()[0])
test_dataset['Exterior1st']=test_dataset['Exterior1st'].fillna(test_dataset['Exterior1st'].mode()[0])
test_dataset['Exterior2nd']=test_dataset['Exterior2nd'].fillna(test_dataset['Exterior2nd'].mode()[0])
test_dataset['BsmtFinType1']=test_dataset['BsmtFinType1'].fillna(test_dataset['BsmtFinType1'].mode()[0])
test_dataset['BsmtFinSF1']=test_dataset['BsmtFinSF1'].fillna(test_dataset['BsmtFinSF1'].mean())
test_dataset['BsmtFinSF2']=test_dataset['BsmtFinSF2'].fillna(test_dataset['BsmtFinSF2'].mean())
test_dataset['BsmtUnfSF']=test_dataset['BsmtUnfSF'].fillna(test_dataset['BsmtUnfSF'].mean())
test_dataset['TotalBsmtSF']=test_dataset['TotalBsmtSF'].fillna(test_dataset['TotalBsmtSF'].mean())
test_dataset['BsmtFullBath']=test_dataset['BsmtFullBath'].fillna(test_dataset['BsmtFullBath'].mode()[0])
test_dataset['BsmtHalfBath']=test_dataset['BsmtHalfBath'].fillna(test_dataset['BsmtHalfBath'].mode()[0])
test_dataset['KitchenQual']=test_dataset['KitchenQual'].fillna(test_dataset['KitchenQual'].mode()[0])
test_dataset['Functional']=test_dataset['Functional'].fillna(test_dataset['Functional'].mode()[0])
test_dataset['GarageCars']=test_dataset['GarageCars'].fillna(test_dataset['GarageCars'].mean())
test_dataset['GarageArea']=test_dataset['GarageArea'].fillna(test_dataset['GarageArea'].mean())
test_dataset['SaleType']=test_dataset['SaleType'].fillna(test_dataset['SaleType'].mode()[0])
print(test_dataset.isnull().sum())
test_dataset.to_csv("ModifyTesthrp.csv")




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:14:15 2017

@author: antfra
"""

#Preprocessing Tasks

#This script is designed to introduce some preprocessing, such as mapping features, encoding class labels, one-hot encoding, data partitioning

#Author: Anthony Franklin

#Date 12/1/17

#Import packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split


#Look at data
Visishop.head()
#Change column names
Visishop.rename(columns = {'Visits per month':'Visits','Visits frequency':'Frequency','Time spent in shop':'Time'}, inplace = True)



#*******************************
#Mapping Ordinal Features
df=pd.DataFrame( [ 
        ['green','M',10.1,'class1' ],
        ['red', 'L', 13.5, 'class2'],
        ['blue', 'XL', 15.3, 'class1']])
df.columns=['color','size','price','classlabel']

#Assume
#XL=L+1=M+2
size_mapping={'XL':3,'L':2,'M':1}
df['size']=df['size'].map(size_mapping)
df

#Reverse map
#inv_size_mapping={v: k for k, v in size_mapping.items()}


#********************************
#Encoding Class Labels

#look at Region Levels
pd.unique(Visishop.Region)

#map region levels to a number (automate using loop)
class_mapping={label: idx for idx,label in enumerate(np.unique(Visishop.Region))}
class_mapping

#transform labels into integers
Visishop.Region=Visishop.Region.map(class_mapping)
Visishop.head()
#Reverse map
#inv_class_mapping={v: k for k, v in class_mapping.items()}

#-or-

#LabelEncoder in scikit-learn
class_le=LabelEncoder()
y=class_le.fit_transform(Visishop.Region.values)
np.unique(y)


#********************************
#OneHotEncoding
ohe=OneHotEncoder(categorical_features=[1])#,sparse=False)
    #Categorical features: what column position for variable to be transformed
    #Sparse=False to return a dense array
X=ohe.fit_transform(Visishop[['Nbr.','Region']]).toarray()
#pd.DataFrame(X).columns

#-or-

#use pandas to create dummy vars                
pd.get_dummies(Visishop.Region)


#include a prefix for dummy var names
dummies = pd.get_dummies(Visishop.Region, prefix='Region_')


df_with_dummy = Visishop[['Region']].join(dummies)
df_with_dummy



#********************************
#Data Partition
#Define X and Y
X,y = Visishop.iloc[:,[0,1,2,4,5]].values, Visishop.iloc[:,3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


#********************************
#Scale
#Normilization
mms=MinMaxScaler()
X_train_norm=mms.fit_transform(X_train[:,[0,2,3]])
X_test_norm=mms.transform(X_test[:,[0,2,3]])

#Standardization
stdsc=StandardScaler()
X_train_std=stdsc.fit_transform(X_train[:,[0,2,3]]) #What went wrong
#X_train_std=stdsc.fit_transform(X_train.iloc[:,1])
X_test_std=stdsc.transform(X_test[:,[0,2,3]])

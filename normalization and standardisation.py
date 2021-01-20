# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:06:03 2021

@author: Kuntinath Noraje
"""

'''
    Transform techniques
    
Types of transformation

1.Standardisation and Normalization
2.scaling to minimum and maximum
3.Scaling to median and Quantile
4.Guassian transformation
  log transformation
  Resiprocal transformation
  Square_root transformation
  BoxCox transformation
    
 
'''

'''
    1.Standardization
    z=(x-mean)/std

'''
import pandas as pd
df=pd.read_csv(r"D:\sabir\python\Datasets\titanic.csv",usecols=['Pclass','Age','Fare','Survived'])

#fill nan with median
df['Age'].fillna(df.Age.median(),inplace=True)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df_transform=scaler.fit_transform(df)

import matplotlib.pyplot as plt
#%matplotlib inline

plt.hist(df_transform[:,3],bins=20)

'''
    2.MinMax Scaling
        transforms values b/t 0-1
        it is mostly used in Deep learning
        it effects on outliers

    X_scaled=(x-x_min)/(_min-x_max)        
''' 
    
from sklearn.preprocessing import MinMaxScaler
minmax=MinMaxScaler()
df_minmax=pd.DataFrame(minmax.fit_transform(df),columns=df.columns)
df_minmax['Fare'].hist()

'''
    3.Scaling to median and Quantile(Robust scaling)
        
    IQR=75th quantile - 25th quantile
    x_scaled=(x-x_median)/IQR
    
    [0,1,2,3,4,5,6,7,8,9]
    9--90-percentile 90% all values in this group less than 9
    1--10%

it is robust(prevents) to the outlier
'''
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df_robust=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
df_robust['Age'].hist()

'''
4.Guassian transformation
    *it data is not normally distributed then we can apply some mathematica calculation
    and convert that data into normal distribution.
    *some  of the ML algorithms like linear_reg, Logistic_reg assumes that
    data is normally distributed.
    *accuracy
    *good performance
    
  log transformation #it works well when data is left or right squed
  Resiprocal transformation
  Square_root transformation
  BoxCox transformation
 
    when have 0 values in data -- log(0)=0 
    so use log1p i.e (log(0+1)=log(1))
    or u can use log(value + 1)
    
    Boxcox must have +ve values
'''
df=pd.read_csv(r"D:\sabir\python\Datasets\titanic.csv",usecols=['Age','Fare','Survived'])
#fill nan
df['Age'].fillna(df.Age.median(),inplace=True)
df.isnull().sum()

#to check data is normally distributed or normally dostributed or not
#Q-Q plot
import scipy.stats as stat
import pylab
def plot_data(df,feature):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    df[feature].hist()
    
    plt.subplot(1,2,2)
    stat.probplot(df[feature],dist='norm',plot=pylab)
    plt.show()

plot_data(df,'Age')
plot_data(df,'Fare')

#logarthmic transformation
import numpy as np
df['log_age']=np.log(df['Age'])
plot_data(df,'log_age')

df['log_fare']=np.log1p(df['Fare'])
plot_data(df,'log_fare')

#2.Reciprocal transformation
df['res_age']=1/df['Age']
plot_data(df,'res_age')

#3.Sqaure root transformation
df['sqrt_age']=df['Age']**0.5
plot_data(df,'sqrt_age')

#4.Exponetial transformation
df['exp_age']=df['Age']*(1/1.2)
plot_data(df,'exp_age')

#5.BoxCox transformation
df['boxcox_age'],parameters=stat.boxcox(df['Age'])
print(parameters)
plot_data(df,'boxcox_age')

df['boxcox_Fare'],parameters=stat.boxcox(df['Fare']+1)
print(parameters)
plot_data(df,'boxcox_Fare')

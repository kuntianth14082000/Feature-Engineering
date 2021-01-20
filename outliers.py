# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 18:18:26 2021

@author: Kuntinath Noraje
"""

#-------------outliers----------
'''
    Outliers

    in some cases outliers are also important(creditcard fraud detection)
    only some of the ML algorithms are impacted by outliers
    so when u keep outliers dont use these king of algorithm
    
    which ML models are very much sensitive to outliers ?
    
    1.Naive bayes                 -- NOT sensitive to outlier
    2.SVM                         --NOT
    3.LinearRegression            --YES
    4.LogisticRegression          --YES
    5.Decision Tree regressor or classifier --NOT
    6.Ensmble (RF,XGBOOST,GB)     --NOT
    7.Kmeans                    --YES
    8.KNN                       --NOT
    9.Heirarchical              --YES
    10.PCA                      --YES (Very sensitive)
    11.Neural Networks          --YES
    12.LDA                      --YES
    13.DBSCAN                   --YES 
    
NOTE : all unsupervised models impacted to outliers
    
'''
import pandas as pd
df=pd.read_csv(r"D:\sabir\python\Datasets\titanic.csv")
df.head()

import seaborn as sns
sns.distplot(df['Age'].fillna(100))#here 100 is outlier

#Remove Outliers for guassian distribution
figure=df.Age.hist(bins=50)
figure.set_title('Age')
figure.set_xlabel('Age')
figure.set_ylabel('No. of passangers')

figure=df.boxplot(column='Age')
df['Age'].describe()

#if data is normally distributed we use this 
#assumming age follows Guassian distribution, we will calculate boundaries which differentiat the outliers
upper_boundary=df['Age'].mean() + 3*df['Age'].std()
lower_boundary=df['Age'].mean() - 3*df['Age'].std()
#it can be do with Z-score


#------in case of feature is skewed------------
figure=df.Fare.hist(bins=50)
figure.set_title('Fare')
figure.set_xlabel('Fare')
figure.set_ylabel('No. of passangers')

figure=df.boxplot(column='Fare')
df.Fare.describe()

#in case of skeved the IQR will work some time or some time not
IQR=df.Fare.quantile(0.75)-df.Fare.quantile(0.25)
lower_bridge=df.Fare.quantile(0.25)-(IQR*3)
upper_bridge=df.Fare.quantile(0.75)+(IQR*3)


#-----------now remove outliers--------------
data=df.copy()
data.loc[data['Age']>73,'Age']=73
data.loc[data['Fare']>100,'Fare']=100

figure=data.Age.hist(bins=50)
figure.set_title('Age')
figure.set_xlabel('Age')
figure.set_ylabel('No. of passangers')
 


#----------Let's check performance using some algorithms-----------

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df[['Age','Fare']].fillna(0),df['Survived'],test_size=0.1)

from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(x_train,y_train)
y_pred=LR.predict(x_test)
y_pred1=LR.predict_proba(x_test)

from sklearn.metrics import accuracy_score,roc_auc_score
print(accuracy_score(y_test,y_pred))#0.7
print(roc_auc_score(y_test,y_pred1[:,1]))#0.7389686337054757


#-------using random forest
from sklearn.ensemble import RandomForestClassifier
LR=RandomForestClassifier()
LR.fit(x_train,y_train)
y_pred=LR.predict(x_test)
y_pred1=LR.predict_proba(x_test)

from sklearn.metrics import accuracy_score,roc_auc_score
print(accuracy_score(y_test,y_pred))#0.655
print(roc_auc_score(y_test,y_pred1[:,1]))#0.693

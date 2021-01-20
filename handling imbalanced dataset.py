# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 20:34:50 2021

@author: Kuntinath Noraje
"""

'''
    ways to handle imbalanced dataset
    1.random forest classifier technique
    
Note: use it at last moment when your model dont work with other techniques 
      under sampling,over sampling.
'''

import pandas as pd
df=pd.read_csv(r"D:\sabir\python\Datasets\Download\archive\creditcard.csv")

#split independent and dependent data
x=df.drop('Class',axis=1)
y=df.Class

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import GridSearchCV

10.0**np.arange(-2,3)

log_class=LogisticRegression()
grid={'c':10.0**np.arange(-2,3),'penalty':['11','12']}
cv=KFold(n_splits=5,random_state=None,shuffle=False)

#spliting train test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=0)

class_weight=({0:1,1:100})
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(class_weight=class_weight)
rfc.fit(x_train,y_train)

y_pred=rfc.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

'''
2.under sampling
    reduse the points of maximum label
    it use only when have less amount of data
    
    Disadvantage
    it looses too much data.
    model accuracy reduces when data is too much imbalanced
'''
from collections import Counter
from imblearn.under_sampling import NearMiss
nm=NearMiss(0.8)
x_train_nm,y_train_nm=nm.fit_sample(x_train,y_train)
print('the number of classes before fit{}'.format(Counter(y_train)))
print('the number of classes after fit{}'.format(Counter(y_train_nm)))

'''
    3.Over sampling
    
    it is wirking good 
'''
from imblearn.over_sampling import RandomOverSampler
os=RandomOverSampler(0.75)
x_train_nm,y_train_nm=os.fit_sample(x_train,y_train)
print('the number of classes before fit{}'.format(Counter(y_train)))
print('the number of classes after fit{}'.format(Counter(y_train_nm)))

y_pred=rfc.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

'''
    4.SMOTE technique
    
    it creates new points based on nearst point of lowest ratio label.
    it will try to create new points of least labels.
'''
from imblearn.combine import SMOTETomek
os=SMOTETomek(0.5)
x_train_nm,y_train_nm=os.fit_sample(x_train,y_train)
print('the number of classes before fit{}'.format(Counter(y_train)))
print('the number of classes after fit{}'.format(Counter(y_train_nm)))

y_pred=rfc.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

'''
    5.ensemble technique
    
'''
from imblearn.ensemble import EasyEnsembleClassifier
easy=EasyEnsembleClassifier()
easy.fit(x_train,y_train)


y_pred=easy.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
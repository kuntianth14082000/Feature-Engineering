# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 11:18:58 2021

@author: Kuntinath Noraje
"""

'''
handle categorical feature

1.one hot encoding

'''

import pandas as pd

df=pd.read_csv(r'D:\sabir\python\Datasets\titanic.csv',usecols=['Sex'])
pd.get_dummies(df,drop_first=True).head()

df=pd.read_csv(r'D:\sabir\python\Datasets\titanic.csv',usecols=['Embarked'])
df['Embarked'].unique()
df.dropna(inplace=True)
pd.get_dummies(df,drop_first=True).head()

#One Hot Encoding  when have many many categorical features
import pandas as pd
df=pd.read_csv(r'D:\sabir\python\Datasets\benze.csv',usecols=['X0','X1','X2','X3','X4','X5','X6'])
df.head()
df['X0'].unique()
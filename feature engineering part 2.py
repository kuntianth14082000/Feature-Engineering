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
for i in df.columns:
    print(len(df[i].unique()))
lst_10=df.X1.value_counts().sort_values(ascending=False).head(10).index
lst_10=list(lst_10)

import numpy as np
for categories in lst_10:
    df[categories]=np.where(df['X1']==categories,1,0)
df[lst_10]
lst_10.append('X1')

'''
#   mean Encoding
'''
df=pd.read_csv(r"D:\sabir\python\Datasets\titanic.csv",usecols=['Survived','Cabin'])
mean_ordinal=df.groupby(['Cabin'])['Survived'].mean().to_dict()
mean_ordinal
df['mean_ordinal_encoding']=df['Cabin'].map(mean_ordinal)
df.head()

# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 08:32:10 2021

@author: Kuntinath Noraje
"""

#handle categorical feature with many categories count frequency encoding
#one hot encoding with many categories
import pandas as pd

#benz dataset
dataset=pd.read_csv(r'D:\sabir\python\Datasets\benze.csv',usecols=['X1','X2'])

#checking how many dummy columns were created after getting dummie columns
pd.get_dummies(dataset).shape

#checkig how many labels are present in each column
len(dataset['X1'].unique())
len(dataset['X2'].unique())

#getting the frequency of each label in columns
df_frequency_map=dataset.X2.value_counts().to_dict()
df_frequency_map2=dataset.X1.value_counts().to_dict()

#now we replacing labels with their frequency of X2 column
dataset.X2=dataset.X2.map(df_frequency_map)
dataset.head()

#------------------------------------------------------
#Label encoding 

import pandas as pd
import datetime

df=pd.read_csv(r'D:\sabir\python\feature_engg,exp_data_analysis\krish naik\Feature engineering\dataset\Qualification_based_ranking.csv')
data1=df.copy()

#order based on degree
degree_map={'PHD':1,'MASTERS':2,'BE':3,'HIGH_SCHOOL':4}
 
data1['degree_ordinal']=df.DEGREE.map(degree_map)

#order based on sallary
data2=df.copy()
data2['rank']='rank'
salary_map={'rank':'1' if data2.SALARY >= 35000 else '0'}
salary=df['SALARY']
def rank(salary):
    if salary >= 35000:
        return 1
    elif salary < 35000 and salary >= 30000:
        return 2
    elif salary < 30000 and salary >20000:
        return 3
    else :
        return 4
di={'rank':rank(salary)}
data2['salary_rank']=data2.SALARY.map(rank(salary))    

#Gaussian transformation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

tit=pd.read_csv(r'D:\sabir\python\Datasets\titanic.csv',usecols=['Age','Fare','Survived'])

#checking null values
tit.isnull().sum()

#now handle these null values
def impute_na(data,variable):
    #function to fill na with a random values
    df=tit.copy()
    
    #random sampling
    df[variable+'_random']=df[variable]
    
    #extract the random sample to fill the na
    random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)
    
    #pandas need to have some index in order to merge dataset
    random_sample.index=df[df[variable].isnull()].index
    df.loc[df[variable].isnull(),variable+'_random']=random_sample
    
    return df[variable+'_random']

tit['Age']=impute_na(data,'Age')

#checking again for null values
tit.isnull().sum()

#now plot Q-Q plot
def dignostic_plot(df,variable):
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    df[variable].hist()
    
    plt.subplot(1,2,2)
    stats.probplot(df[variable],dist='norm',plot=plt)
    plt.show()
    
#for age column
dignostic_plot(tit,'Age')    

#for fare column
dignostic_plot(tit,'Fare')   

#transformation of fare column
tit['log_fare']=np.log(tit['Fare']+1)   
dignostic_plot(tit,'log_fare')   
#logarithmic distribution make good job in making fare variable look gaussian distribution

#reciprocal transformation (invese)
tit['rec_fare']=1/(tit['Fare']+1)
dignostic_plot(tit,'rec_fare')   

#square root transformation
tit['sqrt_fare']=np.sqrt(tit['Fare'])
dignostic_plot(tit,'sqrt_fare')   

#Exponential transfomation
tit['exp_fare']=tit['Fare']**(1/5)
dignostic_plot(tit,'exp_fare')   

#boxcox transformation
tit['boxcox_fare'],param=stats.boxcox(tit.Fare +1)
print('optimal lambda :',param)
dignostic_plot(tit,'boxcox_fare')   


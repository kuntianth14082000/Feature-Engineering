# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:33:16 2021

@author: Kuntinath Noraje
"""
'''
 Handling missing values:
     
1.this is a MCAR (missing compleatly at random)
NOTE : their is no relation b/t missing values and other features

2.Missing data is not at random(MNAR)
NOTE : their is relation b/t missing values and other features

3.missing at random(MAR)
'''
import pandas as pd

dataset=pd.read_csv(r"D:\sabir\python\Datasets\titanic.csv")
dataset.isnull().sum()

#MNAR
import numpy as np
dataset['cabin_null']=np.where(dataset['Cabin'].isnull(),1,0)
dataset['cabin_null'].mean()

dataset.groupby(['Survived'])['Cabin_null'].mean()

'''
#techniques to handle missing values
1.mean /median /mode replacement
2.Random sample imputation
3.capturing nan values with new features
4.end of distribution imputation
5.Arbitaty imputation
6.frequent categories imputatioon
'''

'''
1.mean/median/mode
When should we apply?
it has the assumption that the data missing at random(MNAR)
solve : by replacing the nan with the most occurant values

#Advantages of mean median imputation 
1.Easy to impliment
2.robust to0

#DisAdvantages
1.Change or Distortion in the Original variance
2.Correlation
'''
df=pd.read_csv(r'D:\sabir\python\Datasets\titanic.csv',usecols=['Age','Fare','Survived'])
df.head()

#checking for null values in form of %
df.isnull().mean()

#here we creating a function which will fill nan values with nan values
def impute_nan(df,variable,median):
    df[variable+"_median"]=df[variable].fillna(median)#here we creating a new variable and fill nan with median values
median=df.Age.median()#getting the median value the missing value variable
median    

impute_nan(df,'Age',median)#calling the function
df.head()

#checking the standard diviation b/t original age and after replacing nan
print(df['Age'].std())
print(df['Age_median'].std())

import matplotlib.pyplot as plt
%matplotlib inline

fig=plt.figure()
ax=fig.add_subplot(111)
df['Age'].plot(kind='kde',ax=ax,color='green')
df.Age_median.plot(kind='kde',ax=ax,color='red')
lines, labels=ax.get_legend_handles_labels()
ax.legend(lines,labels,loc='best')

#-------------------------------------------------------------------------
'''
NOTE : when data is missing compleatly at random use mean/median/mode

2.Random sample imptution :
    it takes randomly values from the data set and fill into nan values.
    
    when it is use?
    it assumes that the data is missing compleatly at random.
    
#Advantages
1.easy to implement
2.there is less distortion in the variance

#DisAdvantage
1.Every situation randomness may not work

'''
import pandas as pd

df=pd.read_csv(r'D:\sabir\python\Datasets\titanic.csv',usecols=['Age','Fare','Survived'])
df.head()

df.isnull().sum()
df.isnull().mean()# % of missing values

#taking random values to fill
df['Age'].dropna().sample(df['Age'].isnull().sum(),random_state=0)

#create a function which will take randomly values from data and replace it in missing value
def impute_nan(df,variable,median):
    df[variable+"_median"]=df[variable].fillna(median)#here we creating a new variable and fill nan with median values
    
    df[variable+"_random"]=df[variable]#copying main column into new created column
    #getting random values from the variable to replace with nan
    random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)
    
    #getting indexes of missing values
    random_sample.index=df[df[variable].isnull()].index
    df.loc[df[variable].isnull(),variable+"_random"]=random_sample#filling nan values with random values according to their index
    
median=df.Age.median()#getting median value of the main column
impute_nan(df,'Age',median)#calling the function
df.head()

df.isnull().sum()#checking again for null values

#ploting 
import matplotlib.pyplot as plt
%matplotlib inline

fig=plt.figure()
ax=fig.add_subplot(111)
df['Age'].plot(kind='kde',ax=ax,color='green')
df.Age_random.plot(kind='kde',ax=ax,color='red')
lines, labels=ax.get_legend_handles_labels()
ax.legend(lines,labels,loc='best')

#------------------------------------------------
'''
3.capturing nan values with new features
    it works well when data is not missing randomly.
    it simply captures the null values and fill with median values
    
#Advantages
1.Easy to implement
2.capture's the importance of missing values

#DisAdvantages
1.creating additional feature(Curse of dimentionality)
'''

import pandas as pd
import numpy as np
df=pd.read_csv(r'D:\sabir\python\Datasets\titanic.csv',usecols=['Age','Fare','Survived'])
df.head()

df.isnull().sum()
df.isnull().mean()# % of missing values

df['Age_nan']=np.where(df['Age'].isnull(),1,0)

#-----------------------------------------------------
'''
4.end of distribution imputation

#Advantages
1.Easy to implement
2.capture the importance of missing values

#DisAdvantages
1.Distors the original distribution of the variable
2.

'''
import pandas as pd
import numpy as np
df=pd.read_csv(r'D:\sabir\python\Datasets\titanic.csv',usecols=['Age','Fare','Survived'])
df.head()

df.isnull().sum()
df.isnull().mean()# % of missing values

#firstly take the values from the end of the distribution(>std)
import seaborn as sns
df.Age.hist(bins=50)

df.Age.mean()+3*df.Age.std()

sns.boxplot(df['Age'],data=data)

def impute_nan(df,variable,median,extreme):
    df[variable+"_end_distribution"]=df[variable].fillna(extreme)#here we creating a new variable and fill nan with median values
    df[variable].fillna(median,inplace=True)
    
impute_nan(df,'Age',df.Age.median(),extreme)

#----------------------------------------------------------
'''
5.Arbitary imputation
    it fills nan values with arbitary values.

'''

import pandas as pd

df=pd.read_csv(r'D:\sabir\python\Datasets\titanic.csv',usecols=['Age','Fare','Survived'])
df.head()

df['Age'].hist(bins=50)

#----------------------------------------------------------
'''
6.categorical missing values(Frequent categorical imputation)
    here we replace nan with the most occured value
#advantages
1.Easy to implement
2.Faster way to implement

#DisAdvantages
1.since we are using the most frequent labels, it may use them in an over represented way,if their  are many nans
2.it destorts relationship b/t the most frequent values
'''
import pandas as pd
df=pd.read_csv(r"D:\sabir\python\Datasets\Deep Learning fahad\Advanced house price prediction\train.csv",usecols=['BsmtQual','FireplaceQu','GarageType','SalePrice'])
df.head()
df.columns
df.isnull().sum()
df.shape

#get % of missing values
df.isnull().mean().sort_values(ascending=True)

#compute the frequency with every feature

#for  BsmtQual column
df.groupby(['BsmtQual'])['BsmtQual'].count().sort_values(ascending=True).plot.bar()
#or
df['BsmtQual'].value_counts().plot.bar()

#for GarageType column
df['GarageType'].value_counts().plot.bar()

#for FireplaceQu column
df['FireplaceQu'].value_counts().plot.bar()

#now create a function to replace nan with most frequent values
def impute_nan(df,variable):
    most_frequent_value=df[variable].value_counts().index[0]#here we get most frequent category
    df[variable].fillna(most_frequent_value,inplace=True)
    
#create a loop to fill nan value of all features
for feature in ['BsmtQual','GarageType','FireplaceQu']:
    impute_nan(df,feature)
    
#check for null values again
df.isnull().sum()

'''
2.adding a variable to capture nan in categorical data

Advantages

DisAdvantages
1.create more featers(curse of dimentionality)
'''
import pandas as pd
import numpy as np
df=pd.read_csv(r"D:\sabir\python\Datasets\Deep Learning fahad\Advanced house price prediction\train.csv",usecols=['BsmtQual','FireplaceQu','GarageType','SalePrice'])
df.head()


df['BsmtQual_var']=np.where(df['BsmtQual'].isnull(),1,0)
frequent_value=df['BsmtQual'].mode()[0]#most frequent label
df['BsmtQual'].fillna(frequent_value,inplace=True)

'''
3.suppose u have more frequent categories, we just replace nan with new category
    here we simply replace nan with new label name i.e missing
'''
import pandas as pd
import numpy as np
df=pd.read_csv(r"D:\sabir\python\Datasets\Deep Learning fahad\Advanced house price prediction\train.csv",usecols=['BsmtQual','FireplaceQu','GarageType','SalePrice'])

def impute_nan(df,variable):
    df[variable+'newvar']=np.where(df[variable].isnull(),'missing',df[variable])
    
for feature in ['BsmtQual','GarageType','FireplaceQu']:
    impute_nan(df,feature)
    
df.head()
#after  replacing nan with other label and created new column drop old column
df=df.drop(['BsmtQual','GarageType','FireplaceQu'],axis=1)
df.head()
#we can replace in the same column no problem

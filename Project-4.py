#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


# In[2]:


df=pd.read_csv('HR-Employee.csv')
df.head()


# In[3]:


df.describe()


# In[4]:


df.isnull().sum()


# In[5]:


df.columns


# In[6]:


df.shape


# In[7]:


df['Attrition']=df['Attrition'].apply(lambda x : 1 if x=='Yes' else 0)


# In[8]:


sns.boxplot(x='Attrition',y='Age',data=df)


# In[9]:


sns.boxplot(x='Attrition',y='DailyRate',data=df)


# In[10]:


sns.boxplot(x='Attrition',y='YearsAtCompany',data=df)


# In[11]:


num_cat=['Education','EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','PerformanceRating','RelationshipSatisfaction','WorkLifeBalance']
for i in num_cat:
    df[i]=df[i].astype('category')


# In[12]:


df=pd.get_dummies(df)


# In[13]:


df.info()


# In[14]:


df['Age'].describe()


# In[15]:


del df['Over18_Y']


# In[16]:


df.shape


# In[17]:


X=df[df.columns.difference(['Attrition'])]
y=df['Attrition']


# In[18]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.33,random_state=42)


# In[19]:


numeric_variables = list(df.select_dtypes(include='int64').columns.values)


# In[20]:


numeric_variables.remove('Attrition')


# In[21]:


numeric_variables


# In[22]:


#First is to reset index for X_train and X_test
X_train.reset_index(drop=True,inplace=True)
X_test.reset_index(drop=True,inplace=True)
#Seprate into two dataframes for numeric and non numeric variable
X_train_num=X_train[numeric_variables]
X_train_nnum=X_train[X_train.columns.difference(numeric_variables)]
X_test_num=X_test[numeric_variables]
X_test_nnum=X_test[X_train.columns.difference(numeric_variables)]


# In[23]:


#Set Standard Scaler
scaler = StandardScaler()


# In[ ]:





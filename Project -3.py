#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error#removing space
from sklearn.model_selection import train_test_split


# In[2]:


df=pd.read_csv('avocado.csv')
df.head()


# In[3]:


df


# In[4]:


df.drop(df.columns[0],axis=1,inplace=True)


# In[5]:


df


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


for col in ['AveragePrice','Total Volume', '4046', '4225','4770','Total Bags','Small Bags','Large Bags','XLarge Bags','year','type','region','Date']:
    df[col].fillna(df[col].mode()[0],inplace=True)


# In[9]:


df.isnull().values.any()


# In[10]:


sns.heatmap(df.isnull())


# In[11]:


df.corr()


# In[12]:


corr_hmap=df.corr()
plt.figure(figsize=(8,7))
sns.heatmap(corr_hmap,annot=True)
plt.show()


# In[37]:


df['AveragePrice'].plot.box()


# In[39]:


df['Total Volume'].plot.box()


# In[40]:


df.plot(kind='box',subplots=True,layout=(2,7))


# In[13]:


df.skew()


# In[14]:


df['AveragePrice'].plot.hist()


# In[15]:


df['year'].plot.hist()


# In[16]:


from scipy.stats import boxcox
# 0 -> log transform
# .5 -> Square root transform
df['AveragePrice']=boxcox(df['AveragePrice'],0)


# In[17]:


df['AveragePrice'].plot.hist()


# In[18]:


plt.scatter(df['Total Volume'],df['AveragePrice'])


# In[19]:


plt.scatter(df['year'],df['AveragePrice'])


# In[20]:


plt.scatter(df['Total Bags'],df['AveragePrice'])


# In[21]:


df.shape


# In[22]:


x=df.iloc[:,0:-1]
x.head()


# In[23]:


y=df.iloc[:,-1]
y.head()


# In[24]:


x.shape


# In[25]:


y.shape


# In[26]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33,random_state=42)


# In[27]:


x_train.shape


# In[28]:


y_train.shape


# In[29]:


x_test.shape


# In[30]:


y_test.shape


# In[ ]:





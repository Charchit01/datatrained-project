#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('heartdisease-data.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.dtypes


# In[5]:


df.info()


# # Summary Statistics

# In[6]:


df.describe()


# In[7]:


sns.heatmap(df.isnull())


# In[9]:


df.corr()


# In[11]:


corr_hmap=df.corr()
plt.figure(figsize=(8,7))
sns.heatmap(corr_hmap,annot=True)
plt.show()


# In[13]:


#univariate Analysis
df['63'].plot.box()


# In[14]:


df['2.1'].plot.box()


# In[15]:


df['4'].plot.box()


# In[16]:


#for checking skewness
sns.distplot(df['4'])


# In[17]:


sns.distplot(df['63'])


# In[18]:


#bivariate analysis
plt.scatter(df['2.1'],df['63'])


# In[ ]:





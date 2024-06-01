#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('IRIS (1).csv')


# In[3]:


df.head()


# In[4]:


plt.figure(figsize=(16,7))
sns.countplot(x = df['sepal_length'],hue=df['species'],palette='inferno')


# In[5]:


plt.figure(figsize=(16,7))
sns.countplot(x = df['sepal_width'],hue=df['species'],palette='inferno')


# In[6]:


plt.figure(figsize=(16,7))
sns.countplot(x = df['petal_width'],hue=df['species'],palette='inferno')


# In[7]:


plt.figure(figsize=(16,7))
sns.countplot(x = df['petal_length'],hue=df['species'],palette='inferno')


# In[8]:


plt.figure(figsize=(10,10))
sns.scatterplot(x=df['petal_length'],y=df['petal_width'],data=df,hue='species',s=50)


# In[9]:


plt.figure(figsize=(10,10))
sns.scatterplot(x=df['sepal_length'],y=df['sepal_width'],data=df,hue='species',s=50)


# In[10]:


X = df.drop(['species'],axis=1)
y = df['species']


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)


# In[12]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[13]:


lr.fit(X_train,y_train)
lr.score(X_train, y_train)


# In[14]:


pred = lr.predict(X_test)


# In[15]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)


# In[ ]:





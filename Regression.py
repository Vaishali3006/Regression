#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston


# In[3]:


#understanding dataset
boston = load_boston()
print(boston.DESCR)


# In[4]:


#access data attributes
dataset=boston.data
for name,index in enumerate(boston.feature_names):
    print(index,name)


# In[5]:


#reshaping data
data=dataset[:,12].reshape(-1,1)


# In[6]:


#shape of data
np.shape(dataset)


# In[7]:


#target values
target = boston.target.reshape(-1,1)


# In[8]:


#shape of target
np.shape(target)


# In[9]:


#ensuring the working of matplotlib inside notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data,target,color='green')
plt.xlabel('lower income population')
plt.ylabel('cost of house')
plt.show()


# In[10]:


#regression
from sklearn.linear_model import LinearRegression

#creating regression model
reg=LinearRegression()

#fit the model
reg.fit(data,target)


# In[11]:


#prediction
pred=reg.predict(data)


# In[13]:


#ensuring the working of matplotlib inside notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data,target,color='red')
plt.scatter(data,pred,color='blue')
plt.xlabel('lower income population')
plt.ylabel('cost of house')
plt.show()


# In[16]:


#circunventing curve issue using polynomial model
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline


# In[19]:


model= make_pipeline(PolynomialFeatures(3),reg)


# In[20]:


model.fit(data, target)


# In[21]:


pred= model.predict(data)


# In[22]:


#ensuring the working of matplotlib inside notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data,target,color='red')
plt.scatter(data,pred,color='blue')
plt.xlabel('lower income population')
plt.ylabel('cost of house')
plt.show()


# In[24]:


# r^2 matrix
from sklearn.metrics import r2_score
#working for pushand pull


# In[25]:


#predict
r2_score(pred,target)


# In[ ]:





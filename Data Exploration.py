#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pylab import rcParams
rcParams['figure.figsize'] = 25, 10


# ### Data Analysis of (COVID 19 India Data)

# In[14]:


# Reading csv file
df = pd.read_csv('india_covid19.csv')
df


# In[4]:


###Converting into date format
df['Date'] = pd.to_datetime(df['Date'],format='%d-%m-%Y')


# ### Correleation Among variables

# In[64]:


df.corr().style.background_gradient(cmap='coolwarm')


# In[6]:


### Ploting 
sns.heatmap(df.corr(),cmap="coolwarm")


# In[63]:


cases = df.dropna()
sns.pairplot(cases, size=1.5)


# In[66]:


# adding correlation coefficient to plot
from scipy import stats

def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate('r = {:.2f}'.format(r), xy=(0.1, 1.0), xycoords=ax.transAxes)

pair_plot = sns.pairplot(cases, size=1.5);
pair_plot.map_lower(corrfunc);
pair_plot.map_upper(corrfunc);


# In[8]:


df.describe()


# In[13]:


sns.set_theme(style="whitegrid")
sns.boxplot(x='Total_Confirmed',data=df)


# In[11]:


sns.boxplot(data=df, orient="h", palette="Set2")


# In[8]:


df.head()


# In[9]:


df.tail()


# In[12]:


# Plotting time vs total_confirmed_cases
plt.plot(df.Date,df.Total_Confirmed,color='blue',label='Total_Confirmed')
plt.title('Total Confirmed Cases')


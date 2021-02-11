#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pylab import rcParams
rcParams['figure.figsize'] = 25, 10


# In[2]:


# Reading csv file
df = pd.read_csv('india_covid19.csv')
df


# In[3]:


### Matrix input (Let x be length of y)
matrix = df.iloc[:,[1, 3, 4, 5, 6]]
y = df.iloc[:,[2]]
x = np.arange(len(y))
x


# # Polynomial Regression (to predict Total Confirmed Cases)

# In[4]:


from sklearn.preprocessing import PolynomialFeatures
Poly = PolynomialFeatures(degree=3)
X = Poly.fit_transform(x.reshape(-1,1))


# In[26]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X,y)


# In[27]:


reg.coef_


# In[28]:


reg.intercept_


# In[29]:


y_pred = reg.predict(X)


# In[30]:


plt.scatter(x,y,color='yellow')
plt.plot(x,y_pred,color='r')
plt.legend(["Predicted", "Actual"]) 
plt.title('Polynomial Regression(degree=3)')
plt.xlabel('Number of Days')
plt.ylabel('Total Confirmed Cases')
plt.show()


# In[31]:


### Accuracy of polynomial regression
reg.score(X,y)


# In[32]:


reg.predict(Poly.transform([[1000]]))


# In[33]:


reg.predict(Poly.transform([[950]]))


# # Random Forest Regression to predict total confirmed cases

# In[35]:


x = x.reshape(-1,1)
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor()
reg.fit(x,y)


# In[36]:


### Accuracy of Random Forest
reg.score(x,y)


# In[37]:


plt.scatter(x,y,color='yellow')
plt.plot(x,reg.predict(x),color='r')
plt.legend(["Predicted", "Actual"]) 
plt.title('Random Forest Regression')
plt.xlabel('Number of Days')
plt.ylabel('Total Confirmed Cases')
plt.show()


# In[38]:


### Overfitting
reg.predict([[275]])


# In[39]:


reg.predict([[250]])


# In[40]:


reg.predict([[300]])


# # Support Vector Regression to predict Total Confirmed Cases

# In[41]:


###SVM RBF MODEL
#Standerized x & y
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
Sx = sc_X.fit_transform(x)
Sy = sc_X.fit_transform(y)


# In[42]:


from sklearn.svm import SVR
reg = SVR(kernel='rbf')
reg.fit(Sx,Sy)


# In[43]:


### Accuracy of SVM
reg.score(Sx,Sy)


# In[44]:


plt.scatter(Sx,Sy,color='yellow')
plt.plot(Sx,reg.predict(Sx),color='r')
plt.legend(["Predicted", "Actual"]) 
plt.title('Support Vector Regression')
plt.xlabel('Number of Days(Standard Form)')
plt.ylabel('Total Confirmed Cases(Standard Form)')
plt.show()


# # Naive Bayes Regression to predict total confirmed cases

# In[45]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train,y_train)


# In[46]:


model.score(X_train,y_train)


# In[47]:


model.score(X_test,y_test)


# In[49]:


y_pred = model.predict(x)


# In[50]:


plt.scatter(x,y,color='yellow')
plt.plot(x,y_pred,color='r')
plt.legend(["Predicted", "Actual"]) 
plt.title('Naive Bayes Regression')
plt.xlabel('Number of Days')
plt.ylabel('Total Confirmed Cases')
plt.show()


# In[51]:


## We are dividing total confirmed analysis in 4 parts to get detail analysis of total confirmed cases


# In[52]:


x1 = x[0:55,:]
x2 = x[55:110,:]
x3 = x[110:165,:]
x4 = x[165:,:]
x1.size


# In[53]:


y1 = df.iloc[0:55,[2]]
y2 = df.iloc[55:110,[2]]
y3 = df.iloc[110:165,[2]]
y4 = df.iloc[165:,[2]]


# In[54]:


plt.plot(x1,y1)
plt.plot(x2,y2)
plt.plot(x3,y3)
plt.plot(x4,y4)
plt.legend(["30 Jan to 24 March", "25 March to 18 May","19 May to 12 July","13 July to 7 Sep"]) 
plt.title('Covid-19 Total Confirmed Cases')
plt.xlabel('Number of Days')
plt.ylabel('Cases')
plt.show()


# ### 0 to 54 (55 samples) Polynomial Regression

# In[58]:


from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 5) 
X_poly = poly.fit_transform(x1.reshape(-1,1)) 
  
poly.fit(X_poly, y1) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y1) 


# In[60]:


poly_var_train, poly_var_test, res_train, res_test = train_test_split(X_poly, y1, test_size = 0.3, random_state = 4)
from sklearn.model_selection import train_test_split 
regression = LinearRegression()
model = regression.fit(poly_var_train, res_train)
model.score(poly_var_test, res_test)
### For 2 - 80.7267035%
### For 3 - 94.3587309%
### For 4 - 97.962313%
### For 5 - 99.12423%


# In[55]:


plt.plot(x1,y1,color='red')
plt.scatter(x1,y1,color='blue')
plt.xlabel('No of Days')
plt.ylabel('Total Confirmed Cases')
plt.legend(["Predicted", "Actual"]) 
plt.title('Total Confirmed Cases')
plt.xlabel('Number of Days')
plt.ylabel('Cases')
plt.show()


# ### From 55 to 109(Another 55 Samples)

# In[61]:


from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 2) 
X_poly = poly.fit_transform(x2.reshape(-1,1)) 
  
poly.fit(X_poly, y2) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y2) 


# In[63]:


poly_var_train, poly_var_test, res_train, res_test = train_test_split(X_poly, y2, test_size = 0.3, random_state = 4)
from sklearn.model_selection import train_test_split 
regression = LinearRegression()
model = regression.fit(poly_var_train, res_train)
model.score(poly_var_test, res_test)
### For 2 - 99.3706768%
### For 3 - 99.9324245%
### For 4 - 99.941485%
### For 5 - 99.967608%
### For 6 - 99.974804%


# In[64]:


plt.scatter(x2, y2, color = 'blue') 
plt.plot(x2, lin2.predict(poly.fit_transform(x2)), color = 'red') 
plt.title('Polynomial Regression(degree=2)') 
plt.xlabel('No of Days')
plt.ylabel('Total Confirmed Cases')
plt.legend(["Predicted", "Actual"]) 
plt.xlabel('Number of Days')
plt.show()


# ### From 110 to 164 (Another 55 Samples)

# In[65]:


from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 2) 
X_poly = poly.fit_transform(x3.reshape(-1,1)) 
  
poly.fit(X_poly, y3) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y3) 


# In[66]:


poly_var_train, poly_var_test, res_train, res_test = train_test_split(X_poly, y3, test_size = 0.3, random_state = 4)
from sklearn.model_selection import train_test_split 
regression = LinearRegression()
model = regression.fit(poly_var_train, res_train)
model.score(poly_var_test, res_test)
### For 1 - 93.89156621%
### For 2 - 99.82907711%
### For 3 - 99.99152763%
### For 4 - 99.9934106%
### For 5 - 99.9963232%


# In[68]:


plt.scatter(x3, y3, color = 'blue') 
plt.plot(x3, lin2.predict(poly.fit_transform(x3)), color = 'red') 
plt.title('Polynomial Regression(degree=2)') 
plt.xlabel('No of Days')
plt.ylabel('Total Confirmed Cases')
plt.legend(["Predicted", "Actual"]) 
plt.xlabel('Number of Days')
plt.show()


# ### From 165 to 221 (Another 57 Samples)

# In[69]:


from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 2) 
X_poly = poly.fit_transform(x4.reshape(-1,1)) 
  
poly.fit(X_poly, y4) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y4) 


# In[71]:


poly_var_train, poly_var_test, res_train, res_test = train_test_split(X_poly, y4, test_size = 0.3, random_state = 4)
from sklearn.model_selection import train_test_split 
regression = LinearRegression()
model = regression.fit(poly_var_train, res_train)
model.score(poly_var_test, res_test)
### For 1 - 98.70816258%
### For 2 - 99.9807448%
### For 3 - 99.9810235%
### For 4 - 99.9958782%
### For 5 - 99.9960550%


# In[72]:


plt.scatter(x4, y4, color = 'blue') 
plt.plot(x4, lin2.predict(poly.fit_transform(x4)), color = 'red') 
plt.title('Polynomial Regression(degree=2)') 
plt.xlabel('No of Days (13 July to 7 Sep)')
plt.ylabel('Total Confirmed Cases')
plt.legend(["Predicted", "Actual"]) 
plt.xlabel('Number of Days')
plt.show()


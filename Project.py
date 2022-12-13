#!/usr/bin/env python
# coding: utf-8

# In[153]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score

df = pd.read_csv("FuelConsumption.csv")
df.head()


# In[154]:


X = np.asarray(df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asarray(df['CO2EMISSIONS'])
X = preprocessing.StandardScaler().fit(X).transform(X)


# In[155]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[156]:


LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR


# In[157]:


yhat = LR.predict(X_test)


# In[158]:


yhat_prob = LR.predict_proba(X_test)
yhat_prob


# In[159]:


print (classification_report(y_test, yhat))


# In[160]:


from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)


# In[161]:


from sklearn import linear_model
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','CO2EMISSIONS']]
msk = np.random.rand(len(df)) < 0.8
set_1 = cdf[msk]
set_2 = cdf[~msk]

train_x = np.asanyarray(set_1[['ENGINESIZE']])
train_y = np.asanyarray(set_1[['CO2EMISSIONS']])


regr = linear_model.LinearRegression()



regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_[0][0],'\nIntercept: ',regr.intercept_[0])


# In[162]:


plt.scatter(set_1.ENGINESIZE, set_1.CO2EMISSIONS,  color='blue')
plt.title("ENGINESIZE vs CO2EMISSIONS")
plt.xlabel("ENGINESIZE")
plt.ylabel(r"CO2EMISSIONS")
m, b = np.polyfit(set_1.ENGINESIZE, set_1.CO2EMISSIONS, 1)

plt.plot(set_1.ENGINESIZE, m*set_1.ENGINESIZE+b,"r")
plt.show()


# In[163]:


from sklearn.metrics import mean_squared_error
import math

r = np.corrcoef(set_1.ENGINESIZE, set_1.CO2EMISSIONS)
print(r)

X_test = X_test[0:-1]
MSE = mean_squared_error(X_train, X_test)
 
RMSE = math.sqrt(MSE)
print(RMSE)


# In[94]:


Seldf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','CO2EMISSIONS']]
SeldfHistogram = Seldf[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','CO2EMISSIONS']]

SeldfHistogram.hist(facecolor='g', alpha=0.75)
plt.grid(True)
plt.show()


# In[106]:


cols = ['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','CO2EMISSIONS']
means =[]
for i in cols:
    means.append(df[i].mean())
print(means)


# In[113]:


def mae_loss(theta, y_vals):
    return np.mean(np.abs(y_vals - theta))

thetas = np.arange(-2, 8, 0.05)
y_vals=np.array([3.3462980318650346, 5.794751640112465, 13.296532333645752, 9.47460168697282, 256.2286785379569])
losses = [mae_loss(theta, y_vals) for theta in thetas]

plt.figure(figsize=(4, 2.5))
plt.plot(thetas, losses)
plt.axvline(np.median(y_vals), linestyle='--',
                    label=rf'Median: $\theta = 2$')
plt.title(r'Mean Absolute Error when $\bf{y}$$ = [3.3462980318650346, 5.794751640112465, 13.296532333645752, 9.47460168697282, 256.2286785379569] $')
plt.xlabel(r'$ \theta $ Values')
plt.ylabel('Loss');
plt.legend();


# In[ ]:





# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 05:21:01 2025

@author: Davakhi
"""


#   IMPORT LIBRARY
import numpy as np # array
import matplotlib.pyplot as plt
import pandas as pd

# IMPORT THE DATA SET

dataset = pd.read_csv(r"E:\NareshIT\MARCH\20th- slr\Salary_Data.csv")

# INDEPENCENT VARIABLE
x = dataset.iloc[:,:-1].values
# DEPENCENT VARIABLE
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


x_train = x_train.values.reshape(-1, 1)

x_test = x_test.values.reshape(-1, 1)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test) 

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

plt.scatter(x_test, y_test, color = 'red')  
plt.plot(x_train, regressor.predict(x_train), color = 'blue')  # Regression line from training set
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_train, y_train, color = 'red')  # Real salary data (training)
plt.plot(x_train, regressor.predict(x_train), color = 'blue')  # Predicted regression line
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# ==== best fit line hear ( what next )

coef = print(f"Coefficient: {regressor.coef_}")

intercept = print(f"Intercept: {regressor.intercept_}")

comparison = pd.DataFrame({'Actual':y_test,})

# future prediction code

exp_12_future_pred = 9312 * 100 + 26780
exp_12_future_pred


bias = regressor.score(x_train, y_train)
print(bias)
               
variance = regressor.score(x_test, y_test)
print(variance)

# can we implement statsticc to this dataset 

dataset.mean()
dataset.median()
dataset['Salary'].mean() 
dataset['Salary'].median()
dataset['Salary'].mode()
dataset.var()
dataset['Salary'].var()
dataset.std()
dataset['Salary'].std()

# Coefficient of variation(cv)
# for calculating cv we have to import a library first
from scipy.stats import variation
variation(dataset.values) # this will give cv of entire dataframe 
variation(dataset['Salary']) # this will give us cv of that particular columm

#Correlation
dataset.corr() # this will give correlation of entire dataframe
dataset['Salary'].corr(dataset['YearsExperience'])  # this will give us correlation between these tw
#Skewness
dataset.skew()
dataset['Salary'].skew()
#Standard Error
dataset.sem() # this will give standard error of entire dataf
dataset['Salary'].sem() 
#Z-score
# for calculating Z-score we have to import a library first
import scipy.stats as stats
dataset.apply(stats.zscore) # this will give Z-score of entire dataframe

stats.zscore(dataset['Salary']) # this will give us Z-score of that particular column

# Degree of Freedom
a = dataset.shape[0] # this will gives us no.of rows
b = dataset.shape[1] # this will give us no.of columns
degree_of_freedom = a-b
print(degree_of_freedom) # this will give us degree of freedom for entire dataset

#Sum of Squares Regression (SSR)

#First we have to separate dependent and independent variables
X=dataset.iloc[:,:-1].values #independent variable
y=dataset.iloc[:,1].values # dependent variable
y_mean = np.mean(y) # this will calculate mean of dependent variable
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
y_predict = reg.predict(X_test) # before doing this we have to train,test and split our 
SSR = np.sum((y_predict-y_mean)**2)
print(SSR)

# Sum of Squares Error (SSE)
#First we have to separate dependent and independent variables
X=dataset.iloc[:,:-1].values #independent variable
y=dataset.iloc[:,1].values # dependent variable
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
y_predict = reg.predict(X_test) # before doing this we have to train,test and split our 
y = y[0:6]
SSE = np.sum((y-y_predict)**2)
print(SSE)

#Sum of Squares Total (SST)
mean_total = np.mean(dataset.values) # here df.to_numpy()will convert pandas Dataframe to Nump
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

#R-Square
r_square = SSR/SST
r_square

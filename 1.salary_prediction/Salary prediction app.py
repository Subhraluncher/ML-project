import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\subhra kanta sahoo\OneDrive\Desktop\A VS CODE\1.mlproject\1.salary_prediction\Salary_Data.csv")

print("Dataset Shape:", dataset.shape)  # (30, 2)

x = dataset.iloc[:, :-1]  
y = dataset.iloc[:, -1] 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_train, y_train, color = 'red') 
plt.plot(x_train, regressor.predict(x_train), color = 'blue')  
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test, y_test, color = 'red')  
plt.plot(x_train, regressor.predict(x_train), color = 'blue') 
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

y_20 = m_slope * 20 + c_intercept
y_20

print(f"Intercept: {regressor.intercept_}")
print(f"Coefficient: {regressor.coef_}")

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

dataset.mean()

dataset.std()

dataset.var()

from scipy.stats import variation
variation(dataset.values) 

variation(dataset['Salary'])

dataset.corr()

dataset.skew()

dataset.sem()

import scipy.stats as stats
dataset.apply(stats.zscore)

y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

r_square = 1 - (SSR/SST)
r_square

print (regressor)

bias = regressor.score(x_train,y_train)
print(bias)

variance  = regressor.score(x_test,y_test)
print(variance)










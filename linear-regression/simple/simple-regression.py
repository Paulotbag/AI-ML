#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Salary_Data.csv') #read the csv documnet
x = dataset.iloc[:,:-1].values #assigning data from the csv file into the x variable (multi-array) (features)
y = dataset.iloc[:,-1].values #only assigning the last column (dependent variable)

#splitting dataset between training set and test set
from sklearn.model_selection import train_test_split #usually we proceed with this to split between training and test set
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0) #good practice to leave 20% of the whole data for the test set

#training simple lienar regression model on training set
from sklearn.linear_model import LinearRegression #sklearn has many important mathematical classes and methods, like the LinearRegression, so we use a lot this library for ML and AI
regressor = LinearRegression()
regressor.fit(x_train, y_train) #training regressor according to the training set

#predicting the test set result
y_pred = regressor.predict(x_test) #predict method from the LinearRegression class is used to predict something according to the provided argument. Here we're predicting the salary


#visualising training set results
plt.scatter(x_train, y_train, color = 'red') #scatter is used to put the points on the chart. x_train and y_train are the coordinates. 
plt.plot(x_train, regressor.predict(x_train), color = 'blue') #plot the curve, here will be a line, on the chart.
plt.title('Salary vs Experience (training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show() #show the chart

#visualising the test set results
plt.scatter(x_test, y_test, color = 'red') 
plt.plot(x_train, regressor.predict(x_train), color = 'blue') #regression line will be the same as if repleced by x_test in this case (simple regression)
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#predicting salary from a single value
#at this point, our regressor object is already trained. So, we just need to predict based on a single value
print(regressor.predict([[12]])) #use double squared brackets because "predict" only accepts 2D arrays
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#data encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough') #[3] is the column that has categorical data on which we have to encode. 'passthrough' maintain all other columns.
x = np.array(ct.fit_transform(x))

#splitting dataset between training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0) #remember that 0.2 means that the test set will have 20% of your total data

#training the multiple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #building the multiple linear regression model
regressor.fit(x_train, y_train) #training the model based on the training set

#predicting the test results
y_predict = regressor.predict(x_test)
np.set_printoptions(precision=2) #help print values with only 2 decimal values
print(np.concatenate((y_predict.reshape(len(y_predict),1), y_test.reshape(len(y_test),1)), 1)) #concatenate needs a tuple as the first argument. Because we want to concatenate 2 arrrays and print them vertically, we had to use reshape to make them vertical.

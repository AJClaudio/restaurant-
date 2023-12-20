
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#imported and formatted data
data = pd.read_csv("restaurant_data.csv")
data.fillna(0, inplace= True)
x = data[["INCENTIVE AMOUNT","TOTAL PROJECT COST"]].values
y = data["JOBS CREATED: ASPIRATIONAL"].values
#data split into training and testing
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size= .2)

#linear regression model
model = LinearRegression().fit(xtrain, ytrain)

coef = np.around(model.coef_, 6)
intercept = round(float(model.intercept_), 6)
r_squared = round(model.score(x, y),2)

# print out the linear equation and r^2 value
print(f"Model's Linear Equation: y={coef[0]}x1 + {coef[1]}x2 + {intercept}")
print("R Squared value:", r_squared)

# get the predicted y values for the xtest values - returns an array of the results
predict = model.predict(xtest)
# round the value in the np array to 2 decimal places
predict = np.around(predict, 2)
print(predict)

# compare the actual and predicted values
print("\nTesting Multivariable Model with Testing Data:")
for index in range(len(xtest)):
    actual = ytest[index] # gets the actual y value from the ytest dataset
    predicted_y = predict[index] # gets the predicted y value from the predict variable
    x_coord = xtest[index] # gets the x value from the xtest dataset
    print(f"Incentive Amount: {x_coord[0]} Total Project Cost: {x_coord[1]} Actual: {actual} Predicted: {predicted_y}")


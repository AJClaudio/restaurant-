
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

coef = np.around(model.coef_, 2)
intercept = round(float(model.intercept_), 2)
r_squared = round(model.score(x, y),2)

# print out the linear equation and r^2 value
print(f"Model's Linear Equation: y={coef[0]}x1 + {coef[1]}x2 + {intercept}")
print("R Squared value:", r_squared)


from re import T
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import warnings
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
warnings.filterwarnings("ignore")

data = pd.read_csv(r"Salary_Data.csv")
df = pd.DataFrame(data)
df.drop_duplicates(subset=None,keep='first',inplace=True)

# Created input and output dataframes
X = df.drop(['Salary'], axis=1)
y = df["Salary"]

y2 = df['Salary'].values.reshape(-1, 1)
X2 = df['YearsExperience'].values.reshape(-1, 1)

# Train - Test split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state = 22)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size = 0.20,random_state = 22)



X_train = X_train['YearsExperience']
X_test = X_test['YearsExperience']

m = len(X_train)
n = 2

#print(X_train)


def Weights(m):
    weights = np.zeros(m)
    return weights

#def CostFunction(predictions,label): 
#   costfunction = 0.5*np.sum(np.square(predictions-label))
#    return costfunction

def StochasticGradientDescent(label,alpha,input,m,n,iter):
    theta = Weights(n)
    X = np.array([input.to_numpy()])
    y = label.to_numpy()
    y = y.reshape(m,1)
    X_0 = np.array([np.ones(m)])
    X = np.concatenate((X_0,X),axis=0)
    X_t = X.T
    #print(X_t[0].T)
    #print(np.matmul(X_t[0].T.reshape(n,1),np.matmul(X_t[0],theta)-y[0]).shape)
    theta = theta.reshape(n,1)
    for j in range(m):
        for i in range(iter):
            theta = theta - alpha*((np.matmul(X_t[j].T.reshape(n,1),np.matmul(X_t[j],theta)-y[j])).reshape(n,1))

    #print(theta)
    return theta

def BatchGradientDescent(label,alpha,input,m,n,iter):
    theta = Weights(n)
    X = np.array([input.to_numpy()])
    y = label.to_numpy()
    X_0 = np.array([np.ones(m)])
    X = np.concatenate((X_0,X),axis=0)
    theta = theta.reshape(n,1)
    #print(y.shape)
    #print(X.shape)
    #print(theta.shape)
    #print(theta.reshape(n,1))
    #print((np.matmul(X.T,theta.reshape(n,1))).shape)
    #print(np.matmul(X,np.matmul(X.T,theta.reshape(n,1))-y.reshape(m,1)))
    #print(y)
    for j in range(iter):
        theta = theta - alpha*(np.matmul(X,np.matmul(X.T,theta.reshape(n,1))-y.reshape(m,1)))
    #print(theta)
    return theta

def NormalEquation(input,label,m,n):
    X = np.array([input.to_numpy()])
    y = label.to_numpy()
    y = y.reshape(24,1)
    #print(y.shape)
    X_0 = np.array([np.ones(m)])
    X = np.concatenate((X_0,X),axis=0)
    #theta = theta.reshape(n,1)
    #print(np.matmul(X,X.T))
    #print(np.linalg.inv(np.matmul(X.T,X)))
    theta = np.matmul(np.matmul((np.linalg.inv(np.matmul(X,X.T))),X),y)
    return theta

def Hypothesis(theta,X):
    #print(X)
    x = np.array([X.to_numpy()])
    X_0 = np.array([np.ones(len(X))])
    x = np.concatenate((X_0,x),axis=0)
    hypothesis = np.matmul(x.T,theta)
    return hypothesis



theta1  = StochasticGradientDescent(y_train,0.001,X_train,m,2,10000)
theta2 = NormalEquation(X_train,y_train,m,2)
theta3 = BatchGradientDescent(y_train,0.001,X_train,m,2,10000)
X_test_val = np.array([1,6])
y_predicted1 = Hypothesis(theta1,X_test)
y_predicted2 = Hypothesis(theta2,X_test)
y_predicted3 = Hypothesis(theta3,X_test)
#y_predicted1 = y_predicted1.reshape(1,6)
#y_predicted = y_predicted.tolist()
#print(y_test)
y_test = y_test.to_numpy()
print(y_test)
print("\n")
print(y_predicted1)
print("\n")
print(y_predicted2)
print("\n")
print(y_predicted3)
print("\n \n")

regressor = LinearRegression()
regressor.fit(X_train2, y_train2)
print(regressor.intercept_)
print(regressor.coef_)
y_pred = regressor.predict(X_test2)
df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)








    





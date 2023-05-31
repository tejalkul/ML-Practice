from re import T
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import warnings
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
warnings.filterwarnings("ignore")

data = pd.read_csv(r"diabetes-dataset.csv",index_col=False)
df = pd.DataFrame(data)
df.drop_duplicates(subset=None,keep='first',inplace=True)

# Created input and output dataframes
X = df.drop(['Outcome'], axis=1)
y = df["Outcome"]



#print(X)

# Train - Test split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state = 42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size = 0.20,random_state = 22)
m = len(X_train)
n = 9

# Standardized the data
def Standardize(X_train):
    for column in X_train:
        X_train[column] = (X_train[column] - np.mean(X_train[column]))/ np.std(X_train[column])
    
    return X_train

def Weights(m):
    weights = np.zeros(m)
    return weights

def Sigmoid(z):
    return 1/(1 + np.exp(-z))

def BatchGradientDescent(label,alpha,input,m,n,iter):
    theta = Weights(n)
    X_0 = np.array([np.ones(m)])
    X = np.concatenate((input.T,X_0),axis=0)
    #print(X)
    theta = theta.reshape(n,1)
    #print(X.shape)
    #print(theta.shape)
    for j in range(iter):
        z = np.matmul(X.T,theta)
        h = Sigmoid(z)
        theta = theta - alpha*(np.matmul(X,h-label.reshape(m,1)))
    #print(theta)
    return theta

def StochasticGradientDescent(label,alpha,input,m,n,iter):
    theta = Weights(n)
    X_0 = np.array([np.ones(m)])
    X = np.concatenate((input.T,X_0),axis=0)
    X_t = X.T
    #print(X)
    theta = theta.reshape(n,1)
    z = np.matmul(X_t,theta)
    h = Sigmoid(z)
    #print(X_t[0].T.shape)
    #print((h[0]-y[0]))
    for j in range(m):
        for i in range(iter):
            z[j] = np.matmul(X_t[j],theta)
            h[j] = Sigmoid(z[j])
            theta = theta - alpha*((np.matmul(X_t[j].T.reshape(n,1),(h[j]-label[j]).reshape(1,1))).reshape(n,1))
    return theta


def NewtonMethod(label,input,m,n,iter):
    theta = Weights(n)
    X_0 = np.array([np.ones(m)])
    X = np.concatenate((input.T,X_0),axis=0)
    X_t = X.T
    #print(X_t[0][0])
    #print(X)
    theta = theta.reshape(n,1)
    z = np.matmul(X_t,theta)
    h = Sigmoid(z)
    ones = np.ones(m)
    hessian = np.zeros([n,n])
    #print(hessian)
    #l = np.matmul(label,np.log(h)) + np.matmul(1-label,np.log(1-h))
    #l_der = np.matmul(X,h-label.reshape(m,1))
    #d1 = np.matmul(h,(1-h).T,X,X.T)

            #print(hessian[k][l])
        #print(np.linalg.det(hessian))
        # hessian = (1.0/len(X[j])*np.dot(X_t[j], X[j])*np.diag(h[j])*np.diag(ones[j] - h[j]) ) 
	    #print(hessian)
    for j in range(m):
        for i in range(iter):
            for k in range(n):
                    for l in range(n):
                        hessian[k][l] = X_t[j][k]*X_t[j][l]*np.matmul(h[j],1-h[j])
            print(hessian)
            z[j] = np.matmul(X_t[j],theta)
            h[j] = Sigmoid(z[j])
            theta = theta - np.matmul(np.linalg.inv(hessian),(np.matmul(X_t[j].T.reshape(n,1),(h[j]-label[j]).reshape(1,1))).reshape(n,1))


        
    
    return theta


    
def Hypothesis(theta,X):
    #x = np.array([X.to_numpy()])
    X_0 = np.array([np.ones(len(X))])
    X = np.concatenate((X.T,X_0),axis=0)
    #theta = theta.reshape(n,1)
    z = np.matmul(X.T,theta)
    #print(z)
    h = Sigmoid(z)
    hypothesis = []
    for i in range(len(h)):
        if(h[i]<=0.5):
            hypothesis.append(0)
        else:
            hypothesis.append(1)
    length = len(hypothesis)
    hypothesis = np.array(hypothesis)
    hypothesis = hypothesis.reshape(length,1)
    return hypothesis

# Finding accuracy using formula
def Accuracy(y,y_pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    i = 0
    #y = y_given.tolist()
    #print(y)
    for i in range(len(y)):
        if((y[i] == 1) and (y_pred[i] == 1)):
            tp = tp + 1
        elif((y[i] == 0) and (y_pred[i] == 0)):
            tn = tn + 1
        elif((y[i] == 1) and (y_pred[i] == 0)):
            fn = fn + 1
        elif((y[i] == 0) and (y_pred[i] == 1)):
            fp = fp + 1
    
    #precision = tp/(tp + fp)
    #recall = tp/(tp + fn)
    accuracy = (tp + tn)/(tp + fp + fn + tn)
    #print(accuracy)
    #f1_score = 2*precision*recall/(precision + recall)
    return accuracy

X_train = Standardize(X_train)
X_train = X_train.values
print(X_train)
X_test = Standardize(X_test)
X_test = X_test.values
y_test = y_test.values
y_train = y_train.values
theta1  = BatchGradientDescent(y_train,0.01,X_train,m,n,1000)
theta2 = StochasticGradientDescent(y_train,0.01,X_train,m,n,1000)
#theta3 = NewtonMethod(y_train,X_train,m,n,1000)
y_test = y_test.reshape(len(y_test),1)
#print(y_test)
#print("\n")
y_predicted1 = Hypothesis(theta1,X_test)
y_predicted2 = Hypothesis(theta2,X_test)
#y_predicted3 = Hypothesis(theta3,X_test)
#print(y_predicted2)
df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_predicted1.squeeze()})
print(Accuracy(y_test,y_predicted1))
#print(df_preds)

df_preds2 = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_predicted2.squeeze()})
print(Accuracy(y_test,y_predicted2))
#print(df_preds2)

#df_preds3 = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_predicted3.squeeze()})
#print(Accuracy(y_test,y_predicted3))
#print(df_preds3)


regressor = LogisticRegression()
regressor.fit(X_train2, y_train2)
predictions = regressor.predict(X_test2)
df_preds4 = pd.DataFrame({'Actual': y_test2.squeeze(), 'Predicted': predictions.squeeze()})
#print(df_preds4)
score = regressor.score(X_test2,y_test2)
print(score)






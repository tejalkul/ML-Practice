from re import T
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm

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

def GradientCostFunction(X,y,m,n,C,iter,alpha):
    dw = Weights(n)
    w = Weights(n)
    X_0 = np.array([np.ones(m)])
    X = np.concatenate((X.T,X_0),axis=0)
    #print(X.T[0].shape)
    #print(dw.shape)
    for j in range(iter):
        for i in range(m):
            d = 1 - y[i]*np.dot(X.T[i],w)
            #print(d)
            if(max(0,d)==0):
                dw = dw + w
            else:
                dw = dw + w - C*y[i]*X.T[i]

        w = w - alpha*dw/m
            
    #dw = dw/m
    

    return w

def Hypothesis(X,y,w):
    X_0 = np.array([np.ones(len(X))])
    X = np.concatenate((X.T,X_0),axis=0)
    #print(w)
    hypothesis = []
    #print(len(X))
    for i in range(len(X.T)):
        hypothesis.append(np.sign(np.dot(X.T[i],w)))
    
    hypothesis = np.array(hypothesis)
    hypothesis = hypothesis.reshape(len(X.T),1)

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
        elif((y[i] == -1) and (y_pred[i] == -1)):
            tn = tn + 1
        elif((y[i] == 1) and (y_pred[i] == -1)):
            fn = fn + 1
        elif((y[i] == -1) and (y_pred[i] == 1)):
            fp = fp + 1
    
    #precision = tp/(tp + fp)
    #recall = tp/(tp + fn)
    accuracy = (tp + tn)/(tp + fp + fn + tn)
    #print(accuracy)
    #f1_score = 2*precision*recall/(precision + recall)
    return accuracy


X_train = Standardize(X_train)
X_train = X_train.values
#print(X_train)
X_test = Standardize(X_test)
X_test = X_test.values
y_test = y_test.values
y_train = y_train.values

for i in range(len(y_train)):
    if(y_train[i]==0):
        y_train[i] = -1

#for i in range(len(y_train2)):
#    if(y_train2[i]==0):
#        y_train2[i] = -1

for i in range(len(y_test)):
    if(y_test[i]==0):
        y_test[i] = -1



theta1 = GradientCostFunction(X_train,y_train,m,n,52,10000,0.001)
y_pred1 = Hypothesis(X_test,y_test,theta1)

#print(len(X_test))
#print(len(y_test))

df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred1.squeeze()})
print(df_preds)
print(Accuracy(y_test,y_pred1))

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train2,y_train2)

y_pred2 = classifier.predict(X_test2)

df_preds2 = pd.DataFrame({'Actual': y_test2.squeeze(), 'Predicted': y_pred2.squeeze()})
print(df_preds2)
score = classifier.score(X_test2,y_test2)
print(score)





    



    




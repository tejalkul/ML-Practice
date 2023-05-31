from cmath import nan, pi
from os import remove
from re import T
from typing import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import warnings

from sklearn.naive_bayes import MultinomialNB

#from GaussianDiscriminantAnalysis import prob_x_given_y_0, prob_x_given_y_1

#from logisticRegression import Hypothesis
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
warnings.filterwarnings("ignore")

data = pd.read_csv(r"spam.csv",index_col=False)
df = pd.DataFrame(data)
df.drop_duplicates(subset=None,keep='first',inplace=True)

X = df['v2']
y = df["v1"]

y2 = df['v1'].values.reshape(-1, 1)
X2 = df['v2'].values.reshape(-1, 1)

# Train - Test split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state = 42)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size = 0.20,random_state = 22)

def Probability_y(y,m):
    count = 0
    for i in range(m):
        if(y[i]==1):
            count =count+1
    return count/m

def Probability_word_given_y_0(X,y,m,word,length):
    count1=0
    count2=0

    for i in range(m):
        if(y[i]==0):
            count2 = count2 + 1
            for j in range(len(X[i])):
                if(X[i][j]==word):
                    count1 = count1+1

    return (1+count1)/(count2 + length)

def Probability_word_given_y_1(X,y,m,word,length):
    count1=0
    count2=0

    for i in range(m):
        if(y[i]==1):
            count2 = count2 + 1
            for j in range(len(X[i])):
                if(X[i][j]==word):
                    count1 = count1+1
    #print(count1)
    return (1+count1)/(count2 + length)


    

def Hypothesis(phi,X,dict_prob_0,dict_prob_1):
    hypothesis = np.zeros((len(X),1))
    
    for i in range(len(X)):
        prob_0 = 1
        prob_1 = 1
        for j in range(len(X[i])):
            try:
                #print(dict_prob_0[X[i][j]])
                prob_0 = prob_0*dict_prob_0[X[i][j]]
            except:
                prob_0 = prob_0/len(dict_prob_0)
            try:
                prob_1 = prob_1*dict_prob_1[X[i][j]]
            except:
                prob_1 = prob_1/len(dict_prob_1)
        
        prob_0 = prob_0*(1-phi)
        prob_1 = prob_1*phi

        #print(prob_0)
        #print(prob_1)

        if(prob_0>=prob_1):
            hypothesis[i] = 0
        else:
            hypothesis[i] = 1

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


X_train = X_train.values
X_test = X_test.values
y_test = y_test.values
y_test = y_test.reshape(len(y_test),1)
y_train = y_train.values

for i in range(len(y_train)):
    if(y_train[i]=="ham"):
        y_train[i] = 0
    else:
        y_train[i] = 1

dictionary = []
for i in range(len(X_train)):
    dictionary = dictionary + X_train[i].lower().split()

dictionary = list(set(dictionary))
#print(len(dictionary))
dictionary.sort()
#print(dictionary)
dict = Counter(dictionary)

#print(len(dict))

#print(dictionary[100].isalpha())


for item in list(dict):
    if item.isalpha() == False: 
        del dict[item]
    elif len(item) == 1:
        del dict[item]
#dict = dict.most_common(3000)


#print(dict)
#print(len(dict))

for i in range(len(X_train)):
    X_train[i] = X_train[i].lower().split()


len_test = len(X_test)

for i in range(0,521):
    X_test[i] = X_test[i].lower().split()

for i in range(522,len_test):
    X_test[i-1] = X_test[i].lower().split()

X_test = np.delete(X_test,len_test-1)


prob_0 = []
prob_1 = []

length = len(dict)
#print(length)
m = len(X_train)

for i in list(dict):
    #print(i)
    prob_0.append(Probability_word_given_y_0(X_train,y_train,m,i,length))
    prob_1.append(Probability_word_given_y_1(X_train,y_train,m,i,length))

dict_prob_0 = {}
for word in dict:
    #print(word)
    for prob in prob_0:
        dict_prob_0[word] = prob
        prob_0.remove(prob)
        break

dict_prob_1 = {}
for word in dict:
    for prob in prob_1:
        dict_prob_1[word] = prob
        prob_1.remove(prob)
        break

#print(dict_prob_0)
#print(len(dict_prob_0))
#print(dict_prob_1)

phi = Probability_y(y_train,m)
#print(phi)
y_predicted = Hypothesis(phi,X_test,dict_prob_0,dict_prob_1)
y_test2 = np.zeros((len(y_predicted),1))
for i in range(0,521):
    if(y_test[i]=="ham"):
        y_test2[i] = 0
    else:
        y_test2[i] = 1

for i in range(522,len(y_test)):
    if(y_test[i]=="ham"):
        y_test2[i-1] = 0
    else:
        y_test2[i-1] = 1


#print(len(y_test2))
#print(len(y_predicted))

df_preds = pd.DataFrame({'Actual': y_test2.squeeze(), 'Predicted': y_predicted.squeeze()})
print(df_preds)
print(Accuracy(y_test2,y_predicted))


#model1 = MultinomialNB()
#model1.fit(X_train2,y_train2)

#result1 = model1.predict(X_test2)

#print(result1)












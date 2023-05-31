import numpy as np
import cv2
from imutils import paths
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def createFeatureVector(image,size=(32,32)):
    image = cv2.resize(image,size)
    pixel_list = image.flatten()
    pixel_list = np.append(pixel_list,1)
    return pixel_list

image_paths_train = list(paths.list_images("train"))
image_paths_test = list(paths.list_images("valid"))
images_list_train = []
labels_train = []
images_list_test = []
labels_test = []

for(i,image_path) in enumerate(image_paths_train):
    img = cv2.imread(image_path,1)
    label = image_path.split(os.path.sep)[-1].split(".")[0]

    pixels = createFeatureVector(img)
    images_list_train.append(pixels)
    labels_train.append(label)

X_train = np.array(images_list_train)
y_train = np.array(labels_train)

for(i,image_path) in enumerate(image_paths_test):
    img = cv2.imread(image_path,1)
    label = image_path.split(os.path.sep)[-1].split(".")[0]

    pixels = createFeatureVector(img)
    images_list_test.append(pixels)
    labels_test.append(label)

X_test = np.array(images_list_test)
y_test = np.array(labels_test)

def Weights(m,n):
    weights = np.zeros((n,m))
    return weights


def GradientDescent(X,y,alpha,iter,m,n,N,lamda):
    w = Weights(m,n)
    dw = Weights(m,n)
    #print(np.dot(dw,X[0].T))
    for j in range(iter):
        for i in range(N):
            
            if(y[i]=="dog"):
                d = np.dot(w,X[i].T)[1] - np.dot(w,X[i].T)[0] + 1
            else:
                d = np.dot(w,X[i].T)[0] - np.dot(w,X[i].T)[1] + 1
            #print(d)
            if(max(0,d)==0):
                dw = dw + w
            else:
                if(y[i]=="dog"):
                    dw[0] = dw[0] + 2*lamda*w[0] - X[i]
                else:
                    dw[1] = dw[1] + 2*lamda*w[1] - X[i]
        w = w - alpha*dw/N

    return w

def Hypothesis(X,w):
    Y_pred = []
    for i in range(len(X)):
        d = np.dot(w,X[i].T) 
        if(d[0]>d[1]):
            Y_pred.append("dog")
        else:
            Y_pred.append("cat")
    Y_pred = np.array(Y_pred)
    return Y_pred

# Finding accuracy using formula
def Accuracy(y,y_pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    i = 0
    
    for i in range(len(y)):
        if((y[i] == "dog") and (y_pred[i] == "dog")):
            tp = tp + 1
        elif((y[i] == "cat") and (y_pred[i] == "cat")):
            tn = tn + 1
        elif((y[i] == "dog") and (y_pred[i] == "cat")):
            fn = fn + 1
        elif((y[i] == "cat") and (y_pred[i] == "dog")):
            fp = fp + 1
    
    #precision = tp/(tp + fp)
    #recall = tp/(tp + fn)
    accuracy = (tp + tn)/(tp + fp + fn + tn)
    #print(accuracy)
    #f1_score = 2*precision*recall/(precision + recall)
    return accuracy


w = GradientDescent(X_train,y_train,0.001,10000,3073,2,len(X_train),5)

y_pred = Hypothesis(X_test,w)

df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)
print(Accuracy(y_test,y_pred))




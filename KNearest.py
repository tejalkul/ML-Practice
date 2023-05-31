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





#X_train,X_test,y_train,y_test = train_test_split(images_list, labels, test_size = 0.25,random_state = 42)

Y_pred = []
Y_pred2 = []
Y_pred3 = []
Y_pred4 = []

for i in range(len(X_test)):
    img = X_test[i]
    distance = []
    for j in range(len(X_train)):
        d = np.sum(np.abs(X_test[i]-X_train[j]))
        distance.append(d)
    distance = np.array(distance)
    min_index = np.argmin(distance)
    Y_pred.append(y_train[min_index])

Y_pred = np.array(Y_pred)

for i in range(len(X_test)):
    img = X_test[i]
    distance = []
    for j in  range(len(X_train)):
        d = np.sqrt(np.sum((X_test[i]-X_train[j])*(X_test[i]-X_train[j])))
        distance.append(d)
    distance = np.array(distance)
    min_index = np.argmin(distance)
    Y_pred2.append(y_train[min_index])

Y_pred2 = np.array(Y_pred2)


for i in range(len(X_test)):
    img = X_test[i]
    distance = []
    for j in range(len(X_train)):
        d = np.sum(np.abs(X_test[i]-X_train[j]))
        distance.append(d)
    distance = np.array(distance)
    distance = np.sort(distance)
    count1 = 0
    count2 = 0
    for k in range(0,7):
        if(y_train[k]=="dog"):
            count1 = count1+1
        else:
            count2 = count2+1
    if(count1>count2):
        Y_pred3.append("dog")
    else:
        Y_pred3.append("cat")

Y_pred3 = np.array(Y_pred3)


for i in range(len(X_test)):
    img = X_test[i]
    distance = []
    for j in  range(len(X_train)):
        d = np.sqrt(np.sum((X_test[i]-X_train[j])*(X_test[i]-X_train[j])))
        distance.append(d)
    distance = np.array(distance)
    distance = np.sort(distance)
    count1 = 0
    count2 = 0
    for k in range(0,7):
        if(y_train[k]=="dog"):
            count1 = count1+1
        else:
            count2 = count2+1
    if(count1>count2):
        Y_pred4.append("dog")
    else:
        Y_pred4.append("cat")

Y_pred4 = np.array(Y_pred4)



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

df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': Y_pred.squeeze()})
print(df_preds)
print(Accuracy(y_test,Y_pred))

df_preds2 = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': Y_pred2.squeeze()})
print(df_preds2)
print(Accuracy(y_test,Y_pred2))

df_preds3 = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': Y_pred3.squeeze()})
print(df_preds3)
print(Accuracy(y_test,Y_pred3))

df_preds4 = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': Y_pred4.squeeze()})
print(df_preds4)
print(Accuracy(y_test,Y_pred4))

model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train,y_train)
acc = model.score(X_test,y_test)
print(acc)



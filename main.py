from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import csv
import random as rd
import time

numbers = ['0','1','2','3','4','5','6','7','8','9']

def load_data(PATH):
    with open(PATH) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        images = []
        labels = []
        rowcounter = 0
        for row in csv_reader:
            if(rowcounter == 0):
                rowcounter+=1;
                continue
            images.append(np.array(row[1:len(row)]).astype(np.float))
            labels.append(row[0])
    return(np.array(labels),np.array(images))

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.RdPu):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(numbers))
    plt.xticks(tick_marks, numbers)
    plt.yticks(tick_marks, numbers)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_labels_repartition(labels,labels_array,title = "Histogram",cmap=plt.cm.RdPu):
    repartition = np.empty(len(labels))
    for i,label in enumerate(labels):
        repartition[i] = np.count_nonzero(labels_array == label)
    
    plt.bar([i for i in range(len(labels))],repartition,color=cmap([i/len(labels) for i in range(len(labels))]))
    plt.xticks([i for i in range(len(labels))],labels)
    plt.title(title)
    plt.ylabel('Occurences')
    plt.xlabel('Labels')
    
start = time.time() 
labels_train,images_train = load_data('mnist_train.csv')
end = time.time()
print("-- TRAIN DATA LOADED --")
print("-- LOAD TIME = "+str(end-start)+" s --")

plt.figure()
plot_labels_repartition(numbers,labels_train)
plt.show()

start = time.time()
labels_test,images_test = load_data('mnist_test.csv')
end = time.time()
print("-- TEST DATA LOADED --")
print("-- LOAD TIME = "+str(end-start)+" s --")

plt.figure()
plot_labels_repartition(numbers,labels_test)
plt.show()

clf = svm.SVC(gamma = 'scale')
print("-- MODEL PARAMETERS --")
print(clf)

start = time.time()
clf.fit(images_train,labels_train)
end = time.time()

print("-- TRAINING OK --")
print("-- TRAINING TIME = "+str(end-start)+" s --")

start = time.time()
labels_predict = clf.predict(images_test)
end = time.time()

print("-- PREDICTION OK --")
print("-- PREDICTION TIME = "+str(end-start)+" s --")

matrix = confusion_matrix(labels_test,labels_predict,labels=numbers)

accuracy = clf.score(images_test,labels_test)
print("Overal detection accuracy = "+str(accuracy*100)+"%")

normalized_martix = matrix/matrix.sum(axis=1)

plt.figure()
plot_confusion_matrix(matrix)
plt.show()

plt.figure()
plot_confusion_matrix(normalized_martix, title='Normalized confusion matrix')
plt.show()



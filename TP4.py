from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import csv
import random as rd
import time
import pickle

numbers = ['0','1','2','3','4','5','6','7','8','9']

def load_data(PATH):
    """ load data and corresponding labels from csv file """
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
    """ plot a fancy confusion matrix """
    
    fig, ax = plt.subplots(figsize=(15, 12))
    #Setting the axes with labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(numbers))
    plt.xticks(tick_marks, numbers)
    plt.yticks(tick_marks, numbers)
    plt.tight_layout()
    
    thresh = cm.max() / 4.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, np.round(cm[i, j], 4),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_labels_repartition(labels,labels_array,title = "Histogram",cmap=plt.cm.RdPu):
    """ plot an histogram of label repartition """
   
    repartition = np.empty(len(labels))
    for i,label in enumerate(labels):
        repartition[i] = np.count_nonzero(labels_array == label)
    
    plt.bar([i for i in range(len(labels))],repartition,color=cmap([i/len(labels) for i in range(len(labels))]))
    plt.xticks([i for i in range(len(labels))],labels)
    plt.title(title)
    plt.ylabel('Occurences')
    plt.xlabel('Labels')
    
    
    
# train and display train and test datasets

start = time.time() 
labels_train,images_train = load_data('drive/My Drive/Colab Notebooks/TP4_learning_for_robotics/mnist_train.csv')
end = time.time()
print("-- TRAIN DATA LOADED --")
print("-- LOAD TIME = "+str(end-start)+" s --")

plt.figure()
plot_labels_repartition(numbers,labels_train, "train set")
plt.show()

start = time.time()
labels_test ,images_test = load_data('drive/My Drive/Colab Notebooks/TP4_learning_for_robotics/mnist_test.csv')
end = time.time()
print("-- TEST DATA LOADED --")
print("-- LOAD TIME = "+str(end-start)+" s --")

plt.figure()
plot_labels_repartition(numbers,labels_test, "test set")
plt.show()

# train SVC model 

clf = svm.SVC(gamma = 'scale', verbose=1)
print("-- MODEL PARAMETERS --")
print(clf)

start = time.time()
clf.fit(images_train,labels_train)
end = time.time()

print("-- TRAINING OK --")
print("-- TRAINING TIME = "+str(end-start)+" s --")

#save model and display results

with open('drive/My Drive/Colab Notebooks/TP4_learning_for_robotics/clf', 'wb') as outfile:
    pickle.dump(clf, outfile)
print("-- MODEL SAVED with pickle in file clf --")

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
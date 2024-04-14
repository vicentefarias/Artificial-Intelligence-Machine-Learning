import numpy as np
from math import sqrt
from statistics import mode


def NNClassifier(training, testing, training_labels, testing_labels, k):
    '''Runs the Nearest Neighbor classifier:

    Args:
        training: the subset of data corresponding to the training data as a numpy matrix
        testing:  the subset of data corresponding to the testing data as a numpy matrix
        training_labels: the labels for the training data as a numpy array
        testing_labels: the labels for the testing data as a numpy array
        k: the number of nearest neighbors to use

    This function should do the following:

    - preallocate an array `labels` for the predicted labels of the testing data
    - for each row in the testing data, use knn to predict the label
    - at the end, return what percentagle of labels matched, i.e. how many labels in `labels` matched the label in `testing_labels`
    '''
    # for each point
        # run knn on each point and assign its label into labels
    c = 0
    for i in range(len(testing_labels)):
       a = knn(training, training_labels, testing[i], k)
       predicted = a[0]
       x = testing_labels[i]
       actual = x[0]
       if predicted == actual:
           c+=1
    s = c/len(testing_labels)
    return(s)

def knn(data, data_labels, vector, k):
    '''knn should calculate the nearest neighbor

    data: the numpy array of training data
    data_labels: the numpy array of labels for the training data
    vector: a row from the testing data to calculate nearest neighbors
    k: how many nearest neighbors to find


    This function should find the `k` nearest rows in `data` relative to
    `vector`, and take a vote amongst their labels. Whichever has more (b or m), return
    that value'''
    # preallocate distance array
    distance = np.zeros(len(data_labels))
    
    # for each point in data
        # calculate the distance to vector, store in distance array
    for i in range(len(data_labels)):
        distance[i] =  sqrt(((data[i].astype(float) - vector.astype(float)) ** 2).sum())

    # sort distances, and get indexes to use in data_labels (look at np.argsort)
    indexes = np.argsort(distance)
    
    # take vote amongs top labels
    vote = data_labels[indexes]
    return (vote[:k])

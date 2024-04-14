import numpy as np


def parseCSV(filename):
    '''
    Read the file called "filename", expected to be a csv file.

    Return the data as a list-of-lists
    '''
    # return a list of lists
    ret = []

    with open(filename, 'r') as fp:
        for line in fp:
            ret.append(line.strip().split(','))
    return np.array(ret)

def askConfig():
    '''
    Ask the user for the following config variables:
        filename: the name of the csv file
        testpercentage: percentage of dataset to reserve for testing
        classifier: Which classifier to use in {knn, sklearn-knn, svm}

    Returns the information as a dictionary
    '''
    # return a dictionary
    ret = {}

    # collect variables
    ret["filename"] = input('Please provide a filename: ')
    ret["testpercentage"] = int(input("What percentage of the dataset would you like to put aside for testing? "))
    ret["classifier"] = input("What model would you like to run? (knn, sklearn-knn, svm) ")
    return ret
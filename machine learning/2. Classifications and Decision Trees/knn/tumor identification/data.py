
def postProcessCSV(dataset):
    '''
    Takes the dataset as an N x M numpy array.

    Modifies the dataset to:
        - strip the client ID
        - convert the float columns to be floats instead of strings
    Returns:
        - list of labels (m or b) as 1xN vector
        - rest of data as N x (M-1) matrix of floats
    '''
    # strip first column
    dataset = dataset[:,1:]

    # grab labels
    labels = dataset[:,0]

    # get rest as float
    rest = dataset[:,1:].astype(float)

    return labels, rest


def datasetInfo(data, labels):
    '''
    Takes the dataset and labels vector.

    Returns the following statistics as a dictionary:
        rows: N from above, as an integer
        columns: M from above, as an integer
        benign: Number of benign entries in dataset
        malignant: Number of malignant entries in dataset
    '''
    # return a dictionary
    ret = {}

    ret['rows'] = len(data)
    ret['columns'] = len(data[0])
    ret['benign'] = sum(labels == "B")
    ret['malignant'] = sum(labels == "M")
    return ret


def splitDataset(dataset, test_percentage=20):
    '''
    Takes the dataset as an N x M list of lists.

    Returns 2 subsets of the dataset:
        the first is the testing part, which should be test_percentage percent of N
        the first is the training part, which should be 100-test_percentage percent of N
    '''
    total_length = len(dataset)
    return dataset[:int(total_length*test_percentage/100)], dataset[int(total_length*test_percentage/100):]
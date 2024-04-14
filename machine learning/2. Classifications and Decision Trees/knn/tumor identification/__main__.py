from .advanced import advancedStats, scatterMatrix, correlationHeatmap
from .data import datasetInfo, postProcessCSV, splitDataset
from .models import NNClassifier, SKLearnKnnClassifier, SKLearnSVMClassifier
from .utils import askConfig, parseCSV
import numpy as np


def main():
    again = "y"
    while again == "y":
        # Ask user for configuration variables
        config = askConfig()

        # Print out configuration parameters
        print("Reading filename: {}".format(config["filename"]))

        # Read CSV
        csv = parseCSV(config["filename"])

        # shuffle data
        np.random.shuffle(csv)

        # Do some post processing on the csv
        labels, data = postProcessCSV(csv)

        # Print out table information
        dataset_info = datasetInfo(data, labels)

        # Print basic dataset statistics
        print("Dataset size: {} x {}".format(dataset_info["rows"], dataset_info["columns"]))
        print("Number of benign: {}".format(dataset_info["benign"]))
        print("Number of malignant: {}".format(dataset_info["malignant"]))

        print("\n")

        if input("Would you like to see more advanced stats? (y/n) ") == "y":
            # Print advanced dataset statistics
            advancedStats(data, labels)
            
        if input("Would you like to plot the data? (y/n) ") == "y":
            # Plot scatter matrix using seaborn
            scatterMatrix(data, labels, int(input("How many columns should we plot in the scatter matrix? (5) ")))

            # Plot correlation matrix using pandas and matplotlib
            correlationHeatmap(data)

        print("\n")

        # Split the dataset
        print("Splitting dataset into {}% for training and {}% for testing".format(100 - config["testpercentage"], config["testpercentage"]))
        test, train = splitDataset(data, config["testpercentage"])
        test_labels, train_labels = splitDataset(labels, config["testpercentage"])

        # Print some info about test/train split
        print("Test dataset has {} entries".format(len(test)))
        print("Train dataset has {} entries".format(len(train)))


        # When ready, hit enter to run
        input("\n\nHit enter to run algorithm\n\n")

        # run our knn classifier
        if config["classifier"] == "knn":
            k = int(input("How many nearest neighbors? "))
            print("Running knn classifier...")
            accuracy = NNClassifier(train, test, train_labels, test_labels, k)

        # run the knn classifier from scikit-learn
        elif config["classifier"] == "sklearn-knn":
            k = int(input("How many nearest neighbors? "))
            print("Running sklearn-knn classifier...")
            accuracy = SKLearnKnnClassifier(train, test, train_labels, test_labels, k)

        # run the svm classifier from scikit-learn
        else:
            print("Running sklearn svm classifier...")
            accuracy = SKLearnSVMClassifier(train, test, train_labels, test_labels)

        # print the accuracy
        print("Accuracy: {:.1%}".format(accuracy))

        # repeat whole thing if you want
        again = input("Run again? (y/n) ")

if __name__ == "__main__":
    main()

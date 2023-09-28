# Dataset
from sklearn import datasets
# Data processing
import pandas as pd
import numpy as np


# Standardize the data
from sklearn.preprocessing import StandardScaler
# Modeling 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# Hyperparameter tuning
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score

def load_dataset():
    # Load the breast cancer dataset
    data = datasets.load_breast_cancer()
    # Put the data in pandas dataframe format
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df['target']=data.target
    # Check the data information
    print(df.describe())
    return df
 
def split_dataset(df):
    x_train, x_test, y_train, y_test = train_test_split(df[df.columns.difference(['target'])], df['target'], test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

def normalize_feature(x_train, x_test):
    # Initiate scaler
    sc = StandardScaler()
    # Standardize the training dataset
    x_train_transformed = pd.DataFrame(sc.fit_transform(x_train))
    # Standardized the testing dataset
    x_test_transformed = pd.DataFrame(sc.transform(x_test))
    # Summary statistics after standardization
    return x_train_transformed, x_test_transformed

def SVM(x_train_transformed, x_test_transformed, y_train, y_test):
    svc = SVC()
    params = svc.get_params()
    params_df = pd.DataFrame(params, index=[0])
    # Run model
    svc.fit(x_train_transformed, y_train)
    # Accuracy score
    print(f'The accuracy score of the model is {svc.score(x_test_transformed, y_test):.4f}')
    return svc


def tune_svc_hyperparams(svc, x_train_transformed, x_test_transformed, y_train, y_test):
    # List of C values
    C_range = np.logspace(-1, 1, 3)
    print(f'The list of values for C are {C_range}')
    # List of gamma values
    gamma_range = np.logspace(-1, 1, 3)
    print(f'The list of values for gamma are {gamma_range}')


    # Define the search space
    param_grid = { 
        # Regularization parameter.
        "C": C_range,
        # Kernel type
        "kernel": ['rbf', 'poly'],

        # Gamma is the Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        "gamma": gamma_range.tolist()+['scale', 'auto']
        }
    # Set up score
    scoring = ['accuracy']
    # Set up the k-fold cross-validation
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    # Define grid search
    grid_search = GridSearchCV(estimator=svc, 
                            param_grid=param_grid, 
                            scoring=scoring, 
                            refit='accuracy', 
                            n_jobs=-1, 
                            cv=kfold, 
                            verbose=0)
    # Fit grid search
    grid_result = grid_search.fit(x_train_transformed, y_train)
    # Print grid search summary
    print(grid_result)

        # Print the best accuracy score for the training dataset
    print(f'The best accuracy score for the training dataset is {grid_result.best_score_:.4f}')
    # Print the hyperparameters for the best score
    print(f'The best hyperparameters are {grid_result.best_params_}')
    # Print the best accuracy score for the testing dataset
    print(f'The accuracy score for the testing dataset is {grid_search.score(x_test_transformed, y_test):.4f}')

def main():
    df = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(df)
    x_train_transformed, x_test_transformed =normalize_feature(x_train, x_test)
    svm = SVM(x_train_transformed, x_test_transformed, y_train, y_test)
    tune_svc_hyperparams(svm, x_train_transformed, x_test_transformed, y_train, y_test)

main()
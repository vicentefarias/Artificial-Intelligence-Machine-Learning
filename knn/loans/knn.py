import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def load_dataset():
    dataset = pd.read_csv("loan_data_set.csv")
    dataset.isnull().sum()
    #Filling Gender column by mode
    dataset['Gender']=dataset['Gender'].fillna(dataset['Gender'].mode().values[0])
    #Filling Married column by mode 
    dataset['Married']=dataset['Married'].fillna(dataset['Married'].mode().values[0])
    #Filling Dependents column by mode
    dataset['Dependents']=dataset['Dependents'].fillna(dataset['Dependents'].mode().values[0])
    #Filling Self_Employed column by mode
    dataset['Self_Employed']=dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode().values[0])
    #Filling LoanAmount column by mean
    dataset['LoanAmount']=dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean())
    #Filling Loan_Amount_Term column by mode
    dataset['Loan_Amount_Term']=dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode().values[0] )
    #Filling Credit_History column by mode
    dataset['Credit_History']=dataset['Credit_History'].fillna(dataset['Credit_History'].mode().values[0] )
    dataset.isna().sum()
    dataset.drop('Loan_ID', axis=1, inplace=True)
    dataset.head()
    dataset.info()
    return dataset

def encode_labels(dataset):
    gender = {"Female": 0, "Male": 1}
    yes_no = {'No' : 0,'Yes' : 1}
    dependents = {'0':0,'1':1,'2':2,'3+':3}
    education = {'Not Graduate' : 0, 'Graduate' : 1}
    property = {'Semiurban' : 0, 'Urban' : 1,'Rural' : 2}
    output = {"N": 0, "Y": 1}
    dataset['Gender'] = dataset['Gender'].replace(gender)
    dataset['Married'] = dataset['Married'].replace(yes_no)
    dataset['Dependents'] = dataset['Dependents'].replace(dependents)
    dataset['Education'] = dataset['Education'].replace(education)
    dataset['Self_Employed'] = dataset['Self_Employed'].replace(yes_no)
    dataset['Property_Area'] = dataset['Property_Area'].replace(property)
    dataset['Loan_Status'] = dataset['Loan_Status'].replace(output)
    dataset.head()
    return dataset

def extract_vars(dataset):
    y = dataset['Loan_Status']
    x = dataset.drop(['Loan_Status'], axis=1)
    return x,y

def split_dataset(x,y):
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=38, stratify = y)

def KNNClassifier(x_train, x_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(x_train, y_train)
    #Prediction of test set
    prediction_knn = knn.predict(X=x_test)
    #Print the predicted values
    print("Prediction for test set: {}".format(prediction_knn))
    a = pd.DataFrame({'Actual value': y_test, 'Predicted value': prediction_knn})
    print(a)

def main():
    df = load_dataset()
    encoded_df = encode_labels(df)
    x,y = extract_vars(encoded_df)
    x_train, x_test, y_train, y_test = train_test_split(x,y)
    KNNClassifier(x_train, x_test, y_train, y_test)

main()


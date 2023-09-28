import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics


def load_dataset():
    df = pd.read_csv('housing_prices.csv')
    df.shape
    df.head()
    return df

def plot_dataset(df):
    df.plot(x='SquareFeet', y='SalePrice', style='*')
    plt.title('Square Feet vs Sale Price')
    plt.xlabel('Square Feet')
    plt.ylabel('Sale Price')
    plt.show()

def extract_vars(df):
    x = df.iloc[:, :-1].values
    y = df.iloc[:, 1].values
    return x,y

def split_dataset(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def linear_regression(x_train, y_train):
    lr = LinearRegression().fit(x_train, y_train)
    return lr

def plot_lr(lr, x_train, x_test, y_train, y_test):
    y_pred = lr.predict(x_test)
    plt.scatter(x_train, y_train)
    plt.plot(x_test, y_pred, color='red')
    plt.show()
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df.head()
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


def main():
    df = load_dataset()
    plot_dataset(df)
    x,y = extract_vars(df)
    print(x,y)
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    lr = linear_regression(x_train, y_train)
    plot_lr(lr, x_train, x_test, y_train, y_test)

main()
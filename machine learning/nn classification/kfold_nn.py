import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
le = LabelEncoder()

def load_dataset():
    df = pd.read_csv('gender-classification.csv')
    df.describe()
    return df

def encode_labels(df):
    x = pd.get_dummies(df.iloc[:,:-1])
    y = le.fit_transform(df.iloc[:,-1])
    print(x.head())
    return x,y

def train_model(X_train, X_test, y_train, y_test):
    model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(20), dtype='float32'),
            tf.keras.layers.Dense(units=1024, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(lr=0.0001),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    
    # Callback to reduce learning rate if no improvement in validation loss for certain number of epochs
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-8, verbose=0)
    # Callback to stop training if no improvement in validation loss for certain number of epochs
    early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=0)

    history = model.fit(
        X_train, y_train,
        epochs=1000,
        validation_data=(X_test, y_test),
        callbacks=[reduce_lr, early_stop],
        verbose=0
    )

    tr_loss, tr_acc = model.evaluate(X_train, y_train)
    loss, accuracy = model.evaluate(X_test, y_test)

    return model, history, tr_loss, tr_acc, loss, accuracy

def KFold_nn_classification(df, x, y):
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)

    loss_arr = []
    acc_arr = []
    trloss_arr = []
    tracc_arr = []

    temp_acc = 0

    for train, test in kfold.split(df):
        model, history, trloss_val, tracc_val, loss_val, acc_val = train_model(x.iloc[train], x.iloc[test], y[train], y[test])
        if acc_val > temp_acc:
            print("Model changed")
            temp_acc = acc_val
            model.save('best_model.h5')
            train_index = train
            test_index = test
            best_history = history
    trloss_arr.append(trloss_val)
    tracc_arr.append(tracc_val)
    loss_arr.append(loss_val)
    acc_arr.append(acc_val)

    train_loss, train_acc = model.evaluate(x.iloc[train_index], y[train_index])
    test_loss, test_acc = model.evaluate(x.iloc[test_index], y[test_index])

    print("\n==============================")
    print("Train Accuracy: ", train_acc)
    print("Train Loss: ", train_loss)
    print("==============================")
    print("Test Accuracy: ", test_acc)
    print("Test Loss: ", test_loss)

def main():
    df = load_dataset()
    x,y = encode_labels(df)
    KFold_nn_classification(df, x, y)

main()
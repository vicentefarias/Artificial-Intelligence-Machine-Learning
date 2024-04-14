import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def load_dataset():
    df = pd.read_csv('interests_group.csv')
    df.fillna(0, inplace = True)
    print(df.describe())
    return df

def PCA_KMeans(df):
    x = df.iloc[:,2:]
    # Tranform data onto 2D principal components 
    pca = PCA(2)
    data = pca.fit_transform(x)
    
    # Plot principal component variance
    plt.figure(figsize=(10,10))
    var = np.round(pca.explained_variance_ratio_*100, decimals = 1)
    lbls = ['PC'+str(x) for x in range(1,len(var)+1)]
    plt.bar(x=range(1,len(var)+1), height = var, tick_label = lbls)
    plt.ylabel('Variance')
    plt.show()

    model = KMeans(n_clusters = 6, init = "k-means++")
    label = model.fit_predict(data)
    centers = np.array(model.cluster_centers_)
    plt.figure(figsize=(10,10))
    uniq = np.unique(label)
    for i in uniq:
        plt.scatter(data[label == i , 0] , data[label == i , 1] , label = i)
    plt.scatter(centers[:,0], centers[:,1], marker="x", color='k')
    #This is done to find the centroid for each clusters.
    plt.legend()
    plt.show()

def main():
    df = load_dataset()
    PCA_KMeans(df)

main()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks")


def scatterMatrix(data, labels, count=5):
    '''Use seaborn to produce a pairplot of columns

    data: numpy array of data
    labels: numpy array of labels
    count: number of columns to scatter (larger will result in slower)
    '''
    # convert to dataframe, limit number of columns shown for time reasons
    df = pd.DataFrame(data[:,:count])
    
    # use labels as class
    df['labels'] = labels

    # pairplot
    sns.pairplot(df[[0,1,2,3,4,'labels']],hue='labels')
    
    # show plot
    plt.show()

def correlationHeatmap(data):
    '''Use seaborn to produce a heatmap of the columns' correlation

    data: numpy array of data
    '''
    # convert to dataframe
    df = pd.DataFrame(data)
    
    # heatmap of correlations
    sns.heatmap(df.corr())
    
    # show plot
    plt.show()

import pandas as pd

def advancedStats(data, labels):
    '''Advanced stats should leverage pandas to calculate
    some relevant statistics on the data.

    data: numpy array of data
    labels: numpy array of labels
    '''
    # convert to dataframe
    df = pd.DataFrame(data)

    # print skew and kurtosis for every column
    s=0
    for column in df:
        print('Column '+ str(s) +' statistics:')
        print('Skew:')
        print(df.skew(axis=0)[s])
        print('Kurtosis:')
        print(df.kurtosis(axis=0)[s])
        print('\n')
        s+=1
    
    # assign in labels
    df['labels'] = labels
    print("\n\nDataframe statistics")

    # groupby labels into "benign" and "malignant"
    df.groupby('labels')

    # collect means and standard deviations for columns,
    # grouped by label
    a = df.groupby('labels').get_group('B').mean()
    b = df.groupby('labels').get_group('B').std()
    x = df.groupby('labels').get_group('M').mean()
    y = df.groupby('labels').get_group('M').std()

    # Print mean and stddev for Benign
    print('Benign Stats:')
    print('Mean:')
    print(a)
    print('Standard Deviation:')
    print(b)
    print('\n')

    # Print mean and stddev for Malignant
    print('Malignant Stats:')
    print('Mean:')
    print(x)
    print('Standard Deviation:')
    print(y)
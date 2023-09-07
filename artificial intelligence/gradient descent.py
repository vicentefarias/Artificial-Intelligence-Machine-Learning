import pandas as pd
import numpy as np
import sys

def main():
    '''YOUR CODE GOES HERE'''
    # read csv
    file = open(sys.argv[1], 'r')
    tSet = []
    for i in file:
        tSet.append(i)
    file.close()

    feature1  = []
    feature2 = []
    label = []
    for i in range(len(tSet)):
        tSet[i] = tSet[i].split('\n')[0]
        tSet[i] = tSet[i].split(',')
        feature1.append(int(tSet[i][0]))
        feature2.append(int(tSet[i][1]))
        label.append(int(tSet[i][2]))

    # init weights to zero
    weight_1 = 0
    weight_2 = 0
    b = 0

    # update weights until convergence (no classification errors)
    # write feature weights to output file on each iteration (update of weights)
    n = len(label)
    outfile = '{}'.format(sys.argv[2])
    file = open(outfile, 'w')
    while True:
        numError = 0
        for i in range(n):
            f_x = b + weight_1 * feature1[i] + weight_2 * feature2[i]
            if f_x * label[i] <= 0:
                b += label[i]
                weight_1 += label[i] * feature1[i]
                weight_2 += label[i] * feature2[i] 
                numError += 1
                weights = '{}, {}, {}\n'.format(weight_1, weight_2, b)
                file.write(weights)
        if numError == 0:
            break    
    file.close()
    
    

if __name__ == "__main__":
    """DO NOT MODIFY"""
    main()
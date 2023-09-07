# COMS3203 DISCRETE MATHEMATICS
# CODING ASSIGNMENT 4

# YOUR NAME(s): Vicente Farias
# YOUR UNI(s): vf2272

import matplotlib.pyplot as plt
import numpy as np
import random
import itertools
import time
import math

##### Part 1A #########
'''
Parameters:
n: the index of the row of the pascal triangle

Returns:
list: list of length n+1, where each element corresponds to the required row of Pascal triangle
'''
def pascal_triangle(n):
    list = []
    for i in range(n+1):
        new = []
        if i == 0:
            new = [1]
        if i == 1:
            new = [1, 1]
        if i > 1:
            for j in range(len(list)+1):
                if j == 0:
                    new.append(1)
                if j>0 and j <=(len(list)//2):
                    new.append(list[j-1]+list[j])
                if j > len(list)//2:
                    new.append(new[len(list)-j])
        list = new
                
    return list # list of length n +1 , where each element corresponds to the required row of Pascal triangle


##### Part 1B (1) ##### 
'''
Parameters:
interval: tuple (L, R) indicating the range of integers. Both L & R are inclusive
divisors: list of divisors 

Return:
int: number of integers in the interval that are divisble  
'''
def divisible_atleast_once(interval, divisors):
    # interval: tuple (L, R)
    # divisors: array of divisors
    # WRITE YOUR CODE HERE
    count = 0
    x = []
    for divisor in divisors:
        multiples = []
        leftFactor = int(interval[0]/divisor) + (interval[0]%divisor > 0)
        rightFactor = int(interval[1]/divisor)
        while leftFactor<=rightFactor:
            multiples.append(divisor*leftFactor)
            leftFactor+=1
        x.append(multiples)
    l = len(x)
    y = []
    for i in range(l):
        comb = itertools.combinations(x, i)
        for j in comb:
            y.append(j)
    comb = itertools.combinations(x, l)
    y.append(comb.__next__())
    for k in y:
        if len(k) == 0:
            continue
        elif len(k) == 1:
            count += len(k[0])
        elif len(k) > 1:
            first = k[0]
            numberElem = len(k)
            a = []
            for item in first:
                append = True
                itr = 1
                while itr < numberElem:
                    if item not in k[itr]:
                        append = False
                    itr +=1
                if append ==True:
                    a.append(item)
            if len(k)%2 == 0:
                count -= len(a)
            if len(k)%2 == 1:
                count += len(a)
    return count # count of integes in the  interval that are divisible atleast once by any of the divisors present in the list divisors


##### EXTRA CREDIT: Part 1B (2) #####
##### NOTE: Only uncomment the function signatures below if you are planning to do this part #####################
'''
Parameters:
interval: tuple (L, R) indicating the range of integers. Both L & R are inclusive
divisors: list of divisors 

Return:
int: number of integers in the interval that are divisble  
float: time spent in executing the function
'''
#def check_efficient_runs(interval, divisors):
#    # interval: tuple (L, R)
#    # divisors: array of divisors
#    start = time.time()
#    ans = divisible_atleast_once(interval, divisors)
#    end = time.time()
#    return (ans, end - start)

##### Part 2A (a) #####
'''
Parameters:
n: the number of trials/observations in sample

Returns:
list: list of length 2, where list[0] is the frequency of Tails and list[1] is the frequency of Heads
'''
def large_numbers_coin(n):
    evenCount = 0
    oddCount = 0
    for i in range(n):
        x = np.random.randint(2)
        if x == 0:
            evenCount += 1
        if x == 1:
            oddCount += 1
    return [evenCount, oddCount]# list of experiment results

##### Part 2A (b) #####
'''
Parameters:
n: the number of trials/observations in sample

Returns:
list: list of length 6, where list[0] is the frequency of rolling 1 and list[1] is the frequency of rolling 2, etc...
'''
def large_numbers_dice(n):
    first = 0
    second = 0
    third = 0
    fourth = 0
    fifth = 0
    sixth = 0
    for i in range(n):
        x = np.random.randint(6)
        if x == 0:
            first += 1
        if x == 1:
            second += 1
        if x == 2:
            third += 1
        if x == 3:
            fourth += 1
        if x == 4:
            fifth += 1
        if x == 5:
            sixth += 1
    return [first, second, third, fourth, fifth, sixth]# list of experiment results

##### Part 2A (c) #####
'''
Parameters:
n: the number of trials/observations in sample

Returns:
list: list of length 2, where list[0] is the frequency of Tails and list[1] is the frequency of Heads
'''
def large_numbers_rigged_dice(n):
    first = 0
    second = 0
    third = 0
    fourth = 0
    fifth = 0
    sixth = 0
    for i in range(n):
        x = np.random.randint(8)
        if x == 0 or x == 6:
            first += 1
        if x == 1 or x == 7:
            second += 1
        if x == 2:
            third += 1
        if x == 3:
            fourth += 1
        if x == 4:
            fifth += 1
        if x == 5:
            sixth += 1
    return [first, second, third, fourth, fifth, sixth]

##### Part 2A (d) #####
'''
Parameters:
n: the number of trials/observations in sample

Returns:
none
'''

def large_numbers_histogram(n):
    results = large_numbers_dice(n)
    plt.hist([1,2,3,4,5,6], bins = 40, range = [0, 7], weights = results, align="mid")
    plt.xlabel("Face of the die")
    plt.ylabel("Frequency")
    return # no need to return anything, just plot and screenshot (include in PDF)

##### Part 2A (e) #####
'''
Parameters:
n: the number of trials/observations in sample

Returns:
none
'''
def large_numbers_plot(n):
    mean = []
    x = []
    for i in range(n):
        x.append(i)
        temp = large_numbers_dice(i)
        tempMean = (1*temp[0]+2*temp[1]+3*temp[2]+4*temp[3]+5*temp[4]+6*temp[5])/(i+1)
        mean.append(tempMean)
    plt.plot(x,mean)
    plt.xlabel("Rolls")
    plt.ylabel("Mean")
    return # no need to return anything, just plot and screenshot (include in PDF)


##### EXTRA CREDIT: Part 2B (a) #####
##### NOTE: Only uncomment the function signatures below if you are planning to do this part #####################

'''
Parameters:
n: the number of trials/observations in a sample X_1, ... X_n
k: the number of times we draw from the distribution of averages

Returns:
none
'''
#def clt_uniform(n, k):
    # WRITE YOUR CODE HERE
    # no need to return anything, just plot and screenshot (include in PDF)

##### Part 2B (b) #####
'''
Parameters:
n: the number of trials/observations in sample

Returns:
none
'''
#def clt_gamma(n):
    # WRITE YOUR CODE HERE
    # no need to return anything, just plot and screenshot (include in PDF)
 

### DO NOT TURN IN AN ASSIGNMENT WITH ANYTHING BELOW HERE MODIFIED ###
if __name__ == '__main__':
    print("---------------------------------------")
    print("PART 1A: Nth Row of Pascal Triangle")
    print("---------------------------------------")
    print("Test Case 1 (Row 4): ")
    pascal_student_4 = pascal_triangle(4)
    print("Test Case 1 (Your Answer):", pascal_student_4)
    print("Expected Answer: ", [1, 4, 6, 4, 1])
    print()
    print("Test Case 2 (Row 6): ")
    pascal_student_6 = pascal_triangle(6)
    print("Test Case 2 (Your Answer):", pascal_student_6)
    print("Expected Answer: ", [1, 6, 15, 20, 15, 6, 1])
    print("---------------------------------------")

    print("---------------------------------------")
    print("PART 1B: Principle of Inclusion Exclusion")
    print("---------------------------------------")
    print("Part (a)")
    print("---------------------------------------")
    interval = (1,10)
    divisors = [2,3]
    print("Test Case 1: ", interval, divisors)
    divisible_student_1 = divisible_atleast_once(interval, divisors)
    print("Test Case 1 (Your Answer):", divisible_student_1)
    print("Expected Answer: ", 7)
    print()
    interval = (1956, 9013)
    divisors = [307, 419]
    print("Test Case 2: ", interval, divisors)
    divisble_student_2 = divisible_atleast_once(interval, divisors)
    print("Test Case 2 (Your Answer):", divisble_student_2)
    print("Expected Answer: ", 40)
    print()
    interval = (4014, 6707)
    divisors = [31, 5, 191, 233, 373]
    print("Test Case 3: ", interval, divisors)
    divisble_student_3 = divisible_atleast_once(interval, divisors)
    print("Test Case 3 (Your Answer):", divisble_student_3)
    print("Expected Answer: ", 633)
    print("---------------------------------------")

    # ONLY UNCOMMENT THIS TEST CASE IF YOU ARE PLANNING TO DO THE EXTRA CREDIT
    '''
    print("Part (b)")
    print("---------------------------------------")
    interval = (136611947,842842918)
    divisors = [487, 37, 13, 947, 977, 463, 409, 919, 139, 491, 43, 193, 461, 787, 5, 211]
    print("Test Case 1: ", interval, divisors)
    efficient_1_ans, efficient_1_time = check_efficient_runs(interval, divisors)
    print("Test Case 1 (Your Answer):", efficient_1_ans)
    print("Test Case 1 (Time Spent):", efficient_1_time)
    print("Expected Answer: ", 226443102)
    print()
    '''

    print("---------------------------------------")
    print("PART 2A: Law of Large Numbers")
    print("---------------------------------------")
    print("Part (a)")
    print("---------------------------------------")
    print("Test Case 1 (100 trials): ")
    student_ans_1 = large_numbers_coin(100)
    print("Test Case 1 (Your Answer):", student_ans_1)
    print()
    print("Test Case 2 (500 trials): ")
    student_ans_2 = large_numbers_coin(500)
    print("Test Case 2 (Your Answer):", student_ans_2)
    print("---------------------------------------")

    print("Part (b)")
    print("---------------------------------------")
    print("Test Case 1 (100 trials): ")
    student_ans_1 = large_numbers_dice(100)
    print("Test Case 1 (Your Answer):", student_ans_1)
    print()
    print("Test Case 2 (500 trials): ")
    student_ans_2 = large_numbers_dice(500)
    print("Test Case 2 (Your Answer):", student_ans_2)
    print("---------------------------------------")

    print("Part (c)")
    print("---------------------------------------")
    print("Test Case 1 (100 trials): ")
    student_ans_1 = large_numbers_rigged_dice(100)
    print("Test Case 1 (Your Answer):", student_ans_1)
    print()
    print("Test Case 2 (500 trials): ")
    student_ans_2 = large_numbers_rigged_dice(500)
    print("Test Case 2 (Your Answer):", student_ans_2)
    print("---------------------------------------")

    print("Part (d)")
    print("---------------------------------------")
    print("Plotting 5 trials...(remember to exit out of the pyplot window to continue the program) ")
    large_numbers_histogram(5)
    print()
    print("Plotting 10 trials...(remember to exit out of the pyplot window to continue the program) ")
    large_numbers_histogram(10)
    print("Plotting 50 trials...(remember to exit out of the pyplot window to continue the program) ")
    large_numbers_histogram(50)
    print("Plotting 100 trials...(remember to exit out of the pyplot window to continue the program) ")
    large_numbers_histogram(100)
    print("Plotting 500 trials...(remember to exit out of the pyplot window to continue the program) ")
    large_numbers_histogram(500)

    print("Part (e)")
    print("---------------------------------------")
    print("Plotting line plot for 100000: ")
    large_numbers_plot(100000)

    ## ONLY UNCOMMENT THE TEST CASES BELOW IF YOU ARE DOING THE EXTRA CREDIT ##
    '''
    print("---------------------------------------")
    print("PART 2B: Central Limit Theorem")
    print("---------------------------------------")
    print("Part (a)")
    print("---------------------------------------")
    print("Plotting (n = 5, k = 100)...(remember to exit out of the pyplot window to continue the program) ")
    clt_uniform(5, 100)
    print()
    print("Plotting (n = 10, k = 100)...(remember to exit out of the pyplot window to continue the program) ")
    clt_uniform(10, 100)
    print("Plotting (n = 50, k = 100)...(remember to exit out of the pyplot window to continue the program) ")
    clt_uniform(50, 100)
    print("Plotting (n = 100, k = 100)...(remember to exit out of the pyplot window to continue the program) ")
    clt_uniform(100, 100)
    print("Plotting (n = 5, k = 1000)...(remember to exit out of the pyplot window to continue the program) ")
    clt_uniform(5, 1000)
    print("Plotting (n = 10, k = 1000)...(remember to exit out of the pyplot window to continue the program) ")
    clt_uniform(10, 1000)
    print("Plotting (n = 100, k = 1000)...(remember to exit out of the pyplot window to continue the program) ")
    clt_uniform(100, 1000)
    print("Plotting (n = 1000, k = 1000)...(remember to exit out of the pyplot window to continue the program) ")
    clt_uniform(1000, 1000)
    print("Part (b)")
    print("---------------------------------------")
    print("Plotting (n = 5, k = 100)...(remember to exit out of the pyplot window to continue the program) ")
    clt_gamma(5, 100)
    print()
    print("Plotting (n = 10, k = 100)...(remember to exit out of the pyplot window to continue the program) ")
    clt_gamma(10, 100)
    print("Plotting (n = 50, k = 100)...(remember to exit out of the pyplot window to continue the program) ")
    clt_gamma(50, 100)
    print("Plotting (n = 100, k = 100)...(remember to exit out of the pyplot window to continue the program) ")
    clt_gamma(100, 100)
    print("Plotting (n = 5, k = 1000)...(remember to exit out of the pyplot window to continue the program) ")
    clt_gamma(5, 1000)
    print("Plotting (n = 10, k = 1000)...(remember to exit out of the pyplot window to continue the program) ")
    clt_gamma(10, 1000)
    print("Plotting (n = 100, k = 1000)...(remember to exit out of the pyplot window to continue the program) ")
    clt_gamma(100, 1000)
    print("Plotting (n = 1000, k = 1000)...(remember to exit out of the pyplot window to continue the program) ")
    clt_gamma(1000, 1000)
    '''
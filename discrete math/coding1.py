# COMS3203 DISCRETE MATHEMATICS
# CODING ASSIGNMENT 1

# YOUR NAME(s): Vicente Farias
# YOUR UNI(s): vf2272

'''
Returns the number of vowels in a given string s.

Parameters:
s (string): lowercase string without spaces

Returns:
int: number of vowels
'''
def vowel_counter(s):
    count = 0
    # WRITE YOUR CODE HERE
    for i in s:
        if i=="a" or i=="e" or i=="i" or i=="o" or i=="u":
            count+=1
        elif i=="A" or i=="E" or i=="I" or i=="O" or i=="U":
            count += 1
    return count # int

'''

Implements the 'sometimes y' rule on given string s.

Parameters:
s (string): The target number to generate primes up to.

Returns:
boolean: True/False depending on whether the string has y
int: number of vowels in the string originally (wihout sometimes y rule)
int: number of vowels in the string after sometimes y rule
'''
def sometimes_y(s):
    # WRITE YOUR CODE HERE
    y_in_string = False
    original_count = vowel_counter(s)
    new_count = 0
    for i in s:
        if (i=="y" or i=="Y"):
            y_in_string = True
        if i==s[-1] and (i=="y" or i=="Y"):
            new_count+=1
    new_count += original_count
    return y_in_string, original_count, new_count # boolean, int, int

'''
Returns a list of the number of vowels in each word in a sentence.

Parameters:
sentence (string): A string of a sentence.

Returns:
list: a list of the number of vowels for each word in the sentence.
'''
def sentence_counter(sentence):
    # WRITE YOUR CODE HERE
    list = sentence.split()
    counts = []
    for word in list:
        count = sometimes_y(word)[2]
        counts.append(count)
    return counts # list

'''
Returns an an integer that is the nth Fibonacci number.

Parameters:
n (int): The nth Fibonacci number you want.

Returns:
int: the nth fibonacci number.
'''
def recursive_fib(n):
    # WRITE YOUR CODE HERE
    fib_n = 0
    if n==1:
        fib_n = 1
    elif n==0:
        fib_n = 0
    else: 
        fib_n = recursive_fib(n-1)+recursive_fib(n-2)
    return fib_n # int

'''
Returns an an integer that is the nth Fibonacci number.

Parameters:
n (int): The nth Fibonacci number you want.

Returns:
int: the nth fibonacci number.
'''
def iterative_fib(n):
    # WRITE YOUR CODE HERE
    previous = 0
    current = 0
    fib_n = 0
    for i in range(0, n):
        if i==0:
            current = 0
            fib_n = current
        if i==1 or n==1:
            previous = current
            current = 1
            fib_n = current + previous
        else:
            previous = current
            current = fib_n
            fib_n = current + previous
    return fib_n
'''
Returns whether two sentences are synonyms or not, given a list of synonyms.

Parameters:
synonyms (list): A list of tuples of the synonyms you should store.
sentences (tuple): A 2-tuple containing two sentences you want to compare.

Returns:
boolean: Whether the sentences are synonyms or not.
'''
def synonym_checker(synonyms, sentences):
    # WRITE YOUR CODE HERE
    is_synonym = True
    no_period_sentence1 = sentences[0].split('.')
    no_period_sentence2 = sentences[1].split('.')
    words_in_sentence1 = no_period_sentence1[0].split(' ')
    words_in_sentence2 = no_period_sentence2[0].split(' ')
    word_count = 0
    positions = []
    if len(words_in_sentence1) != len(words_in_sentence2):
        return(False)
    for word in words_in_sentence1:
        if word != words_in_sentence2[word_count]:
            positions.append(word_count)
        word_count+=1
    words = []
    for position in positions:
        new = (words_in_sentence1[position],words_in_sentence2[position])
        words.append(new)
    if len(words)!=len(synonyms):
        return(False)
    x = sorted(synonyms)
    a = []
    b = []
    for i in x:
        u = sorted(i)
        a.append(u)
    for i in words:
        v = sorted(i)
        b.append(v)
    c = 0
    for i in a:
        if i !=b[c]:
            return False
        c+=1
    return is_synonym # boolean

######################################################################
### DO NOT TURN IN AN ASSIGNMENT WITH ANYTHING BELOW HERE MODIFIED ###
######################################################################
if __name__ == '__main__':
    print("#######################################")
    print("Welcome to Coding 1: Python Introduction!")
    print("#######################################")
    print()

    print("---------------------------------------")
    print("PART A: Vowel Counting")
    print("---------------------------------------")
    vowel_tests = [['abcdef', 'abcdefy', 'abc def y'], ['cat', 'catty', 'The big cat.'], ['dog', 'ydog', 'I love dogs!']]
    vowel_answers = [[2, (True, 2, 3), [1, 1, 1]], [1, (True, 1, 2), [1, 1, 1]], [1, (True, 1, 1), [1, 2, 1]]]
    for count, test in enumerate(vowel_tests):
        if(vowel_answers[count][0] == vowel_counter(test[0]) and
        vowel_answers[count][1] == sometimes_y(test[1]) and
        vowel_answers[count][2] == sentence_counter(test[2])):
            passed = "PASSED!"
        else:
            passed = "FAILED!"

        print("Test #{}: {}".format(count + 1, passed))
        print("Vowel Count (Correct): ", vowel_answers[count][0])
        print("Vowel Count (Your Answer): ", vowel_counter(test[0]))
        print("Vowel Count with y (Correct): ", vowel_answers[count][1])
        print("Vowel Count with y (Your Answer): ", sometimes_y(test[1]))
        print("Sentence Count (Correct): ", vowel_answers[count][2])
        print("Sentence Count (Your Answer): ", sentence_counter(test[2]))

    print("---------------------------------------")
    print("PART B: Fibonacci")
    print("---------------------------------------")
    tests = [[1, 1], [4, 4], [10, 10]]
    answers = [[1, 1], [3, 3], [55, 55]]
    for count, test in enumerate(tests):
        if(answers[count][0] == recursive_fib(test[0]) and
            answers[count][1] == iterative_fib(test[1])):
            passed = "PASSED!"
        else:
            passed = "FAILED!"

        print("Test #{}: {}".format(count + 1, passed))
        print("Recursive Fibonacci (Correct): ", answers[count][0])
        print("Recursive Fibonacci (Your Answer): ", recursive_fib(test[0]))
        print("Iterative Fibonacci (Correct): ", answers[count][1])
        print("Iterative Fibonacci (Your Answer): ", iterative_fib(test[1]))


    print("---------------------------------------")
    print("PART C: Synonym Checker")
    print("---------------------------------------")
    tests = [
        [[("movie", "film"), ("reviews", "ratings")], ("I heard that movie got good ratings.", "I heard that film got good reviews.")],
        [[("movie", "film")], ("I heard that movie got good ratings.", "I heard that film got good reviews.")],
        [[("movie", "film"), ("reviews", "ratings")], ("I heard that work of cinema got good ratings.", "I heard that film got good reviews.")]
    ]
    answers = [True, False, False]
    for count, test in enumerate(tests):
        if(answers[count] == synonym_checker(test[0], test[1])):
            passed = "PASSED!"
        else:
            passed = "FAILED!"

        print("Test #{}: {}".format(count + 1, passed))
        print("Synonyms:", test[0])
        print("Sentences:", test[1])
        print("Synonym? (Correct): ", answers[count])
        print("Synonym? (Your Answer): ", synonym_checker(test[0], test[1]))



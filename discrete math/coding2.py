# COMS3203 DISCRETE MATHEMATICS
# CODING ASSIGNMENT 2
#
# Before submitting the file to gradescope make sure of the following:
# 1. The name of the file is coding2.py 
# 2. Nothing below the line `if __name__="__main__":` is changed 
# 3. Make sure there are no indentation errors and that the code compiles on your
#    end
#
# YOUR NAME: Vicente Farias
# YOUR UNI: vf2272

import itertools

'''
Returns the proposition, formatted in string form.

Parameters:
prop (list): proposition in nested list form

Returns:
string: 'prop' in string form
'''
def format_prop(prop):
    # BASE CASE: #####################################
    if 1 == len(prop):
        return prop[0]
    ##################################################

    # UNARY OPERATOR (not): ##########################
    if 2 == len(prop):
        # the following two variable declarations are missing LHS #
        op = prop[0] # missing LHS
        b = prop[1] # missing LHS

        if "not" == op:
            formatted_prop = "(" + op + " " + format_prop(b)+ ")"
            return formatted_prop
        else:
            raise ValueError("Unary proposition is not not.")
    ##################################################

    # BINARY OPERATOR (and, or, if, iff, xor): #######
    elif 3 == len(prop):
        # the following three variable declarations are missing LHS #
        op = prop[0] # missing LHS
        a = prop[1] # missing LHS
        b = prop[2] # missing LHS

        if op not in ("if","iff","or","and","xor"):
            raise ValueError("Binary proposition does not have valid connectives.")

        # change "if" and "iff" representation
        if "if" == op:
            op = "->"
        elif "iff" == op:
            op = "<->"

        # format left and right sides of a binary operation
        left_prop = format_prop(a)
        right_prop = format_prop(b)
        formatted_prop = "("+ left_prop + " "+ op +" "+ right_prop +")"
        return formatted_prop
    ####################################################

    # INVALID LENGTH ####################################
    else:
        raise ValueError("Proposition incorrect length.")
    #####################################################

'''
Returns the evaluation (True or False) as an int (1 or 0) of the proposition,
given a proposition in list form and a list of values for each atomic variable.

Parameters:
prop (list): proposition in nested list form.
values (list): list of integers, either 0 or 1 indicating False or True for
each atomic variable in the proposition. 

Returns:
int: 0 for False, 1 for True
'''
def eval_prop(prop, values):
    # BASE CASE: #####################################
    if 1 == len(prop):
        a = prop[0]
        atomic_prop_id =  int(a[1])-1
        return values[atomic_prop_id]
    ##################################################

    # UNARY OPERATOR (not): ##########################
    elif 2 == len(prop):
        # the following two variable declarations are missing LHS #
        op = prop[0] # missing LHS
        a = prop[1] # missing LHS
        val = eval_prop(a, values)
        
        if "not" == op:
            return int(not(val))
        else:
            raise ValueError("Unary proposition is not not.")
    ##################################################

    # BINARY OPERATOR (and, or, if, iff, xor): #######
    elif 3 == len(prop):
        # the following three variable declarations are missing LHS #
        op = prop[0] # missing LHS
        a = prop[1] # missing LHS
        b = prop[2] # missing LHS

        if op not in ("if", "iff", "or", "and", "xor"):
            raise ValueError("Binary proposition does not have valid connectives.")

        # evaluate left and right sides of a binary operation
        left = eval_prop(a, values)
        right = eval_prop(b, values)

        # the line here is an example. fill in the rest.
        if "and" == op:
            return int(left and right)
        elif "if" == op:
            return int(not(left) or right)
        elif "iff" == op:
            return int((left and right) or (not(left) and not(right)))
        elif "or":
            return int(left or right)
        elif "xor":
            return int((left and not right) or (not left and right))

    # INVALID LENGTH ####################################
    else:
        raise ValueError("Proposition incorrect length.")
    #####################################################

'''
Prints a truth table given a proposition in nested list form and 
an integer defining the number of atomic variables. 

Parameters:
prop (list): proposition in nested list form.
n_var (int): the number of atomic variables in prop.  

Returns:
None
'''
def print_table(prop, n_var):
    '''
    fill in here. you will have to use eval_prop and format_prop,
    and will probably have to use the itertools package (already
    imported for you).
    '''
    x = "| "
    for i in range(n_var):
        x+= "p"+str(i+1)+" | "
    y = format_prop(prop)
    x += y 
    print(x)
    for i in range(2**n_var):
        a = bin(i)[2:]
        if len(a) != n_var:
            while len(a)!=n_var:
                a = "0"+a
        x = "| "
        values = []
        for i in a:
            x += i+"  | "
            values.append(int(i))
        m = eval_prop(prop, values)
        x += str(m) 
        print(x)
    pass


if __name__ == '__main__':
    print("---------------------------------------")
    print("Coding Assignment 2: Propositional Logic")
    print("---------------------------------------")

    print()
    values = [1]
    prop = ["not", ["p1"]]
    ps_str = " ".join("p{}={}".format(i + 1, v) for i, v in enumerate(values))
    print("Evaluating proposition p =", format_prop(prop))
    prop_val = eval_prop(prop, values)
    print("over", ps_str, ":", prop_val)

    print()
    values = [1, 1]
    prop = ["and", ["p1"], ["p2"]]
    ps_str = " ".join("p{}={}".format(i + 1, v) for i, v in enumerate(values))
    print("Evaluating proposition p =", format_prop(prop))
    prop_val = eval_prop(prop, values)
    print("over", ps_str, ":", prop_val)

    print()
    values = [1, 0]
    prop = ["iff", ["p1"],["p2"]]
    ps_str = " ".join("p{}={}".format(i + 1, v) for i, v in enumerate(values))
    print("Evaluating proposition p =", format_prop(prop))
    prop_val = eval_prop(prop, values)
    print("over", ps_str, ":", prop_val)

    print()
    values = [1, 1, 0]
    prop = ["if", ["and", ["p1"], ["not", ["p2"]]], ["p3"]]
    ps_str = " ".join("p{}={}".format(i + 1, v) for i, v in enumerate(values))
    prop_str = format_prop(prop)
    print("Evaluating proposition p =", prop_str)
    prop_val = eval_prop(prop, values)
    print("over", ps_str, ":", prop_val)

    print()
    values = [1, 0, 1]
    prop = ["iff", ["p1"], ["or", ["p2"], ["not", ["p3"]]]]
    ps_str = " ".join("p{}={}".format(i + 1, v) for i, v in enumerate(values))
    print("Evaluating proposition p =", format_prop(prop))
    prop_val = eval_prop(prop, values)
    print("over", ps_str, ":", prop_val)

    print("---------------------------------------------------")
    print("Table:")
    print_table(["if", ["and", ["p1"], ["not", ["p2"]]], ["p3"]], 3)
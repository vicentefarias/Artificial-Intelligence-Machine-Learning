#!/usr/bin/env python
#coding:utf-8

"""
Each sudoku board is represented as a dictionary with string keys and
int values.
e.g. my_board['A1'] = 8
"""
import sys

ROW = "ABCDEFGHI"
COL = "123456789"


def print_board(board):
    """Helper function to print board in a square."""
    print("-----------------")
    for i in ROW:
        row = ''
        for j in COL:
            row += (str(board[i + j]) + " ")
        print(row)


def board_to_string(board):
    """Helper function to convert board dictionary to string for writing."""
    ordered_vals = []
    for r in ROW:
        for c in COL:
            ordered_vals.append(str(board[r + c]))
    return ''.join(ordered_vals)


def backtracking(board):
    """Takes a board and returns solved board."""
    # TODO: implement this
    solved_board = backtrack({}, board)
    return solved_board

def backtrack(assignment, board):
    # if no assignment init assignment
    if len(assignment) == 0:
        assignment = initAssignment(board)
    # if complete assignment return assignment
    if complete(assignment):
        return assignment
    # select key with min values remaining
    key, vals = key_min_rem_vals(assignment, board)
    # if no such key or no vals remaining return False
    if key == None or len(vals) == 0:
        return False
    # check each possible value for key with min vals
    # if value is invalid, remove value from assignment 
    # and check next value. If no such value return false
    for val in vals:
        assignment[key] = val
        result = backtrack(assignment, board)
        if result != False:
            return result
        del assignment[key]
    return False

def initAssignment(board):
    assign = {}
    # if board value neq 0
    # add key and value to assignment dictionary
    for k,v in board.items():
        if v != 0:
            assign[k] = v
    return assign

def complete(assignment):
    # Preliminary length check
    if len(assignment) != 81:
        return False
    # Check ROW Constraint
    for r in ROW:
        row = []
        for k, v in assignment.items():
            if r in k:
                row.append(v)
            if len(row) == 9:
                break
        rSet = set(row)
        if len(row) == len(rSet) and rSet == {1,2,3,4,5,6,7,8,9}:
            continue
        else:
            return False

    # Check COL Constraint
    for c in COL:
        col = []
        for k, v in assignment.items():
            if c in k:
                col.append(v)
            if len(col) == 9:
                break
        cSet = set(col)
        if len(col) == len(cSet) and cSet == {1,2,3,4,5,6,7,8,9}:
            continue
        else:
            return False

    # Check BOX Constraint
    boxVals = []
    for i in range(9):
        boxVals.append([])
    for k,v in assignment.items():
        if ('A' in k or 'B' in k or 'C' in k) and ('1' in k or '2' in k or '3' in k):
            boxVals[0].append(v)
        if ('A' in k or 'B' in k or 'C' in k) and ('4' in k or '5' in k or '6' in k):
            boxVals[1].append(v)
        if ('A' in k or 'B' in k or 'C' in k) and ('7' in k or '8' in k or '9' in k):
            boxVals[2].append(v)
        if ('D' in k or 'E' in k or 'F' in k) and ('1' in k or '2' in k or '3' in k):
            boxVals[3].append(v)
        if ('D' in k or 'E' in k or 'F' in k) and ('4' in k or '5' in k or '6' in k):
            boxVals[4].append(v)
        if ('D' in k or 'E' in k or 'F' in k) and ('7' in k or '8' in k or '9' in k):
            boxVals[5].append(v)
        if ('G' in k or 'H' in k or 'I' in k) and ('1' in k or '2' in k or '3' in k):
            boxVals[6].append(v)
        if ('G' in k or 'H' in k or 'I' in k) and ('4' in k or '5' in k or '6' in k):
            boxVals[7].append(v)
        if ('G' in k or 'H' in k or 'I' in k) and ('7' in k or '8' in k or '9' in k):
            boxVals[8].append(v)
    for bVals in boxVals:
        boxSet = set(bVals)
        if len(boxSet) == len(bVals) and boxSet == {1,2,3,4,5,6,7,8,9}:
            continue
        else:
            return False 
    
    return True


def key_min_rem_vals(assignment, board):
    rem_Vals = {}
    # Initalize all unassigned keys with domain
    unassignedKeys = set(board.keys()) - set(assignment.keys())
    for uk in unassignedKeys:
        rem_Vals[uk] = [1,2,3,4,5,6,7,8,9]

    # Apply domain constraints to all unassigned keys
    # using assigned values
    for ak, av in assignment.items():
        for uk, uv in rem_Vals.items():
            if ak[0] in uk and av in uv:
                uv.remove(av)
            if ak[1] in uk and av in uv:
                uv.remove(av)
            if sameBox(ak, uk) and av in uv:
                uv.remove(av)
        
    
    count = 0
    minVals = []
    minKey = ''
    # check if minKey exists
    if len(unassignedKeys) == 0:
        return None, []
    
    # select key with min remaining values
    for k, v in rem_Vals.items():
        if count == 0 or len(v) < minLen:
            minKey = k
            minVals = v
            minLen = len(v)
        count += 1

    return minKey, minVals
            
    
def sameBox(key1, key2):
    if ('A' in key1 or 'B' in key1 or 'C' in key1) and ('A' in key2 or 'B' in key2 or 'C' in key2):
        if int(key1[1]) <= 3 and int(key2[1]) <= 3:
            return True
        if int(key1[1]) > 3 and int(key1[1]) < 7 and int(key2[1]) > 3 and int(key2[1]) < 7:
            return True
        if int(key1[1]) >= 7 and int(key2[1]) >= 7:
            return True
    if ('D' in key1 or 'E' in key1 or 'F' in key1) and ('D' in key2 or 'E' in key2 or 'F' in key2):
        if int(key1[1]) <= 3 and int(key2[1]) <= 3:
            return True
        if int(key1[1]) > 3 and int(key1[1]) < 7 and int(key2[1]) > 3 and int(key2[1]) < 7:
            return True
        if int(key1[1]) >= 7 and int(key2[1]) >= 7:
           return True
    if ('G' in key1 or 'H' in key1 or 'I' in key1) and ('G' in key2 or 'H' in key2 or 'I' in key2):
        if int(key1[1]) <= 3 and int(key2[1]) <= 3:
            return True
        if int(key1[1]) > 3 and int(key1[1]) < 7 and int(key2[1]) > 3 and int(key2[1]) < 7:
            return True
        if int(key1[1]) >= 7 and int(key2[1]) >= 7:
           return True
 
if __name__ == '__main__':
    if len(sys.argv) > 1:
        
        # Running sudoku solver with one board $python3 sudoku.py <input_string>.
        print(sys.argv[1])
        # Parse boards to dict representation, scanning board L to R, Up to Down
        board = { ROW[r] + COL[c]: int(sys.argv[1][9*r+c])
                  for r in range(9) for c in range(9)}       
        
        solved_board = backtracking(board)
        
        # Write board to file
        out_filename = 'output.txt'
        outfile = open(out_filename, "w")
        outfile.write(board_to_string(solved_board))
        outfile.write('\n')

    else:
        # Running sudoku solver for boards in sudokus_start.txt $python3 sudoku.py

        #  Read boards from source.
        src_filename = 'sudokus_start.txt'
        try:
            srcfile = open(src_filename, "r")
            sudoku_list = srcfile.read()
        except:
            print("Error reading the sudoku file %s" % src_filename)
            exit()

        # Setup output file
        out_filename = 'output.txt'
        outfile = open(out_filename, "w")

        # Solve each board using backtracking
        for line in sudoku_list.split("\n"):

            if len(line) < 9:
                continue

            # Parse boards to dict representation, scanning board L to R, Up to Down
            board = { ROW[r] + COL[c]: int(line[9*r+c])
                      for r in range(9) for c in range(9)}

            # Print starting board. TODO: Comment this out when timing runs.
            print_board(board)

            # Solve with backtracking
            solved_board = backtracking(board)

            # Print solved board. TODO: Comment this out when timing runs.
            print_board(solved_board)

            # Write board to file
            outfile.write(board_to_string(solved_board))
            outfile.write('\n')

        print("Finishing all boards in file.")
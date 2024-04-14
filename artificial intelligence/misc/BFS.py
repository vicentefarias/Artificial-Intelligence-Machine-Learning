from __future__ import division
from __future__ import print_function

import sys
import math
import time
import queue as Q


#### SKELETON CODE ####
## The Class that Represents the Puzzle
import resource
class PuzzleState(object):
    """
        The PuzzleState stores a board configuration and implements
        movement instructions to generate valid children.
    """
    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        """
        :param config->List : Represents the n*n board, for e.g. [0,1,2,3,4,5,6,7,8] represents the goal state.
        :param n->int : Size of the board
        :param parent->PuzzleState
        :param action->string
        :param cost->int
        """
        if n*n != len(config) or n < 2:
            raise Exception("The length of config is not correct!")
        if set(config) != set(range(n*n)):
            raise Exception("Config contains invalid/duplicate entries : ", config)

        self.n        = n
        self.cost     = cost
        self.parent   = parent
        self.action   = action
        self.config   = config
        self.children = []

        # Get the index and (row, col) of empty block
        self.blank_index = self.config.index(0)

    def display(self):
        """ Display this Puzzle state as a n*n board """
        for i in range(self.n):
            print(self.config[3*i : 3*(i+1)])

    def move_up(self):
        """ 
        Moves the blank tile one row up.
        :return a PuzzleState with the new configuration
        """
        i0 = self.config.index(0)
        newConfig = [] + self.config
        if i0 >= self.n:
            newConfig[i0], newConfig[i0 - self.n] = self.config[i0 - self.n], self.config[i0]
            return PuzzleState(newConfig, 3)

    def move_down(self):
        """
        Moves the blank tile one row down.
        :return a PuzzleState with the new configuration
        """
        i0 = self.config.index(0)
        newConfig = [] + self.config
        if i0 < (self.n -1) * self.n:
            newConfig[i0], newConfig[i0 + self.n] = self.config[i0 + self.n], self.config[i0]
            return PuzzleState(newConfig, 3)
      
    def move_left(self):
        """
        Moves the blank tile one column to the left.
        :return a PuzzleState with the new configuration
        """
        i0 = self.config.index(0)
        newConfig = [] + self.config
        if i0 % self.n != 0:
            newConfig[i0], newConfig[i0 - 1] = self.config[i0 - 1], self.config[i0] 
            return PuzzleState(newConfig, 3)

    def move_right(self):
        """
        Moves the blank tile one column to the right.
        :return a PuzzleState with the new configuration
        """
        i0 = self.config.index(0)
        newConfig = [] + self.config
        if i0 % self.n != self.n - 1:
            newConfig[i0], newConfig[i0 + 1] = self.config[i0 + 1], self.config[i0]  
            return PuzzleState(newConfig, 3)
    
    def expand(self):
        """ Generate the child nodes of this node """
        
        # Node has already been expanded
        if len(self.children) != 0:
            return self.children
        
        # Add child nodes in order of UDLR
        children = [
            self.move_up(),
            self.move_down(),
            self.move_left(),
            self.move_right()]

        # Compose self.children of all non-None children states
        self.children = [state for state in children if state is not None]
        return self.children

def writeOutput(path, cost, node, maxDepth, rt, maxRam):
    ### Student Code Goes here
    with open('output.txt', 'w') as f:
        f.write("path_to_goal: ")
        f.write("[")
        for i in range(len(path)):
            f.write("'")
            f.write(path[i])
            f.write("'")
            if i != len(path)-1:
                f.write(",]\n")
        f.write("cost_of_path: " + str(cost) + "\n")
        f.write("nodes_expanded: " + str(node)+ "\n")
        f.write("search_depth: " + str(cost) + "\n") 
        f.write("max_search_depth: "+ str(maxDepth) + "\n")
        f.write("running_time: "+ str(rt)[0:8] + "\n")
        f.write("max_ram_usage: "+ str(maxRam) + '\n')
        f.close

def bfs_search(initial_state):
    """BFS search"""
    ### STUDENT CODE GOES HERE ###
    start_time = time.monotonic()
    frontier = [initial_state]
    explored = set()
    count = 0

    while len(frontier) > 0:
        state = frontier.pop(0)
        explored.add(tuple(state.config))

        if test_goal(state):
            temp = state
            cost = 0
            path = []
            while tuple(temp.config) != tuple(initial_state.config):
                if temp.parent != None:
                    path.append(direction(temp.parent, temp))
                temp = temp.parent
                cost += 1
            end_time = time.monotonic()
            ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000
            writeOutput(path[::-1], cost, count, 4, end_time - start_time, ram)
            return state 

        children = state.expand()
        for child in children:
            if child not in frontier and tuple(child.config) not in explored:
                child.parent = state
                frontier.append(child)
        count += 1

    return False


def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    ### STUDENT CODE GOES HERE ###
    for i in range(puzzle_state.n ** 2):
        if puzzle_state.config[i] != i:
            return False
    return True


def direction(state0, state1):
    dir = {}
    up = state0.move_up()
    down = state0.move_down()
    left = state0.move_left()
    right = state0.move_right()
    if up != None:
        dir[tuple(up.config)] = "Up"
    if down != None:
        dir[tuple(down.config)] = "Down"
    if left != None:
        dir[tuple(left.config)] = "Left"
    if right != None:
        dir[tuple(right.config)] = "Right"
    for k in dir.keys():
        if k == tuple(state1.config):
            return dir[k]  

# Main Function that reads in Input and Runs BFS
def main():
    search_mode = "bfs"
    begin_state = sys.argv[1].split(",")
    begin_state = list(map(int, begin_state))
    board_size  = int(math.sqrt(len(begin_state)))
    hard_state  = PuzzleState(begin_state, board_size)
    start_time  = time.time()
    if search_mode == "bfs": bfs_search(hard_state)
    else: 
        print("Enter valid command arguments !")
        
    end_time = time.time()
    print("Program completed in %.3f second(s)"%(end_time-start_time))

if __name__ == '__main__':
    main()
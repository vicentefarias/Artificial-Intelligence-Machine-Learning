import numpy as np

def p1(k: int) -> str:
    product = 1
    ret = '1'
    for i in range(k):
        product = product * (i + 1)
        if i > 0:
            ret = str(product) + "," + ret
    return ret


def p2_a(x: list, y: list) -> list:
    z = sorted(y)
    z = z[::-1]
    z.pop()
    return z


def p2_b(x: list, y: list) -> list:
    z = x[::-1]
    return z


def p2_c(x: list, y: list) -> list:
    z = sorted(list(set(x + y)))
    return z


def p2_d(x: list, y: list) -> list:
    return [x, y]

def p3_a(x: set, y: set, z: set) -> set:
    w = x | y | z
    return w


def p3_b(x: set, y: set, z: set) -> set:
    w = x & y & z
    return w


def p3_c(x: set, y: set, z: set) -> set:
    xy = x & y
    xz = x & z
    yz = y & z
    intersect = xy | xz | yz
    t = x - intersect | y - intersect | z - intersect
    return t

def p4_a() -> np.array:
    arr = np.array([[1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 2, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1]
                    ])
    return arr


def p4_b(x: np.array) -> list:
    m = len(x)
    n = len(x[0])
    a = 0
    b = 0 
    pawn = [0, 0]
    ret = []
    for i in range(m):
        for j in range(n):
            if x[i][j] == 2:
                pawn = [i, j] 
    possAttack = [[pawn[0] - 1, pawn[1] + 2],
                  [pawn[0] - 1, pawn[1] - 2],
                  [pawn[0] + 1, pawn[1] + 2],
                  [pawn[0] + 1, pawn[1] - 2],
                  [pawn[0] - 2, pawn[1] + 1],
                  [pawn[0] - 2, pawn[1] - 1],
                  [pawn[0] + 2, pawn[1] + 1],
                  [pawn[0] + 2, pawn[1] - 1]
                 ]
    for a in possAttack:
        if (0 <= a[0] <= m) and (0 <= a[1] <= n):
            if x[a[0], a[1]] == 1:
                ret.append([a[0], a[1]])
    return ret


def p5_a(x: dict) -> int:
    isolated = 0
    for key, value in x.items():
        if len(value) == 0:
            isolated += 1
    return isolated



def p5_b(x: dict) -> int:
    nonisolated = 0
    for key, value in x.items():
        if len(value) > 0:
            nonisolated += 1
    return nonisolated


def p5_c(x: dict) -> list:
    edges = []
    for key, values in x.items():
        for value in values:
            if [key, value] not in edges:
                edges.append([value, key])
    return list(edges)


def p5_d(x: dict) -> np.array:
    l = len(x)
    arr = np.zeros((l, l))
    count = 0
    values = {}
    for key in x:
        values[key] = count
        count += 1
    for key, value in x.items():
        for v in value:
            x = values[key]
            y = values[v]
            arr[x][y] = 1
    return arr 



class PriorityQueue(object):
    def __init__(self):
        self.prices = {  "apple" : 5.0, 
                    "banana" : 4.5, 
                    "carrot" : 3.3,
                    "kiwi" : 7.4,
                    "orange" : 5.0,  
                    "mango" : 9.1, 
                    "pineapple" : 9.1 }
        self.queue = []

    def push(self, x):    
        if x in list(self.prices.keys()):
            self.queue.append(x)
        pass

    def pop(self):
        if not (self.is_empty()):
            item = self.queue[0]
            max = self.prices[self.queue[0]]
            for i in range(len(self.queue) - 1):
                if max < self.prices[self.queue[i + 1]]:
                    item = self.queue[i+1]
                    max = self.prices[self.queue[i + 1]]
            self.queue.remove(item)
        return item 


    def is_empty(self):
        if (len(self.queue) == 0):
            return True



if __name__ == '__main__':
    print(p1(k=8))
    print('-----------------------------')
    print(p2_a(x=[], y=[1, 3, 5]))
    print(p2_b(x=[2, 4, 6], y=[]))
    print(p2_c(x=[1, 3, 5, 7], y=[1, 2, 5, 6]))
    print(p2_d(x=[1, 3, 5, 7], y=[1, 2, 5, 6]))
    print('------------------------------')
    print(p3_a(x={1, 3, 5, 7}, y={1, 2, 5, 6}, z={7, 8, 9, 1}))
    print(p3_b(x={1, 3, 5, 7}, y={1, 2, 5, 6}, z={7, 8, 9, 1}))
    print(p3_c(x={1, 3, 5, 7}, y={1, 2, 5, 6}, z={7, 8, 9, 1}))
    print('------------------------------')
    print(p4_a())
    print(p4_b(p4_a()))
    print('------------------------------')
    graph = {
        'A': ['D', 'E'],
        'B': ['E', 'F'],
        'C': ['E'],
        'D': ['A', 'E'],
        'E': ['A', 'B', 'C', 'D'],
        'F': ['B'],
        'G': []
    }
    print(p5_a(graph))
    print(p5_b(graph))
    print(p5_c(graph))
    print(p5_d(graph))
    print('------------------------------')
    pq = PriorityQueue()
    pq.push('apple')
    pq.push('kiwi')
    pq.push('orange')
    while not pq.is_empty():
        print(pq.pop())
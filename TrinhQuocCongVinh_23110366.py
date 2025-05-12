import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from collections import deque
import heapq
import time
import random
import math
import copy
import itertools
from copy import deepcopy
from collections import defaultdict


# example_state = [
#     [2, 6, 5],
#     [8, 0, 7], 
#     [4, 3, 1]
# ]
# example_state =[
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 0, 8]
# ]
example_state2 =  [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
example_state =[[1, 2, 3], [4, 6, 0], [7, 5, 8]]


expansion_state = 0
caculation_time = 0
def solve_command(func):
    def wrapper():
        initial_state = get_initial_state()
        initial_state1 = get_initial_state1()
        if initial_state is None:
            return 
        
        global solution,solution2, present_res
        present_res = 0 
        
        if func.__name__ == 'bfs_two_states':
            path = bfs_two_states(initial_state,initial_state1,goal_state)
            if path != None:
                solution = aftermove(initial_state,path)
                solution2 = aftermove(initial_state1,path)
            else:
                solution = None
        else:
            if func.__name__== "Partially":
                solution,solution2 = Partially(initial_state,initial_state1,goal_state)
            else:
                if func.__name__=="KiemThu":
                    solution = KiemThu()
                    solution2 = None
                else:
                    if func.__name__=="backtrack":
                        solution = backtrack()
                        solution2 = None
                    else:
                        if func.__name__=="ac3":
                            solution = ac3()
                            solution2 = None

                        else:
                            solution = func(initial_state, goal_state)
                            solution2 = None
            
        if solution:
            result_text.set("Đã tìm thấy lời giải! Xem các bước bên dưới.")
            messagebox.showinfo("Thông báo",f"Duyệt qua {expansion_state} trạng thái :: Thời gian {round(caculation_time,2)} s \n Kết quả lưu ở file Result.txt")
            with open("Result.txt", "w",encoding= 'utf-8') as f:
                f.write("Các bước giải: \n \n")
                for idx, state in enumerate(solution):
                    f.write(f"Bước {idx}:\n")
                    for row in state:
                        f.write(" ".join(str(num) for num in row) + "\n")
                    f.write("\n")

            update_display()
        else:
            result_text.set("Không tìm thấy lời giải!")
    return wrapper

def next_state(state):
    neighbors = []
    x, y = np.where(state == 0)
    x, y = x[0], y[0]
    moves = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    
    for nx, ny in moves:
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = state.copy()
            new_state[x, y], new_state[nx, ny] = new_state[nx, ny], new_state[x, y]
            neighbors.append(new_state)
    return neighbors
def next_state_cost(state):
    neighbors = []
    x, y = np.where(state == 0)
    x, y = x[0], y[0]
    moves = [(x-1, y, 1), (x+1, y,1), (x, y-1,2), (x, y+1,2)]
    
    for nx, ny  ,cost in moves:
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = state.copy()
            new_state[x, y], new_state[nx, ny] = new_state[nx, ny], new_state[x, y]
            neighbors.append((new_state,cost))
    return neighbors
def bfs(initial, goal):
    global caculation_time,expansion_state
    expansion_state = 0
    start_time = time.time()
    queue = deque([(initial.tolist(), [])])
    visited = set()
    
    while queue:
        state, path = queue.popleft()
        state_tuple = tuple(map(tuple, state))
        
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        expansion_state +=1
        if np.array_equal(state, goal):
            end_time = time.time()
            caculation_time =end_time - start_time
            return path + [state] 
        
        for neighbor in next_state(np.array(state)):
            queue.append((neighbor.tolist(), path + [state]))
    end_time = time.time()
    caculation_time =end_time - start_time
    return None
def dfs(initial, goal):
    global caculation_time,expansion_state
    expansion_state = 0
    start_time = time.time()
    stack = deque([(initial.tolist(), [])])
    visited = set()
    
    while stack:
        state, path = stack.pop()
        state_tuple = tuple(map(tuple, state))
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        expansion_state +=1
        if np.array_equal(state, goal):
            end_time = time.time()
            caculation_time =end_time - start_time
            return path + [state] 
        
        for neighbor in next_state(np.array(state)):
            stack.append((neighbor.tolist(), path + [state]))
    end_time = time.time()
    caculation_time =end_time - start_time
    return None

def ucs(initial, goal):
    global caculation_time,expansion_state
    expansion_state = 0
    start_time = time.time()
    pq = []
    heapq.heappush(pq, (0, initial.tolist(), [])) 
    visited = set()
    
    while pq:
        cost, state, path = heapq.heappop(pq)
        state_tuple = tuple(map(tuple, state))

        
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        expansion_state +=1
        if np.array_equal(state, goal):
            end_time = time.time()
            caculation_time =end_time - start_time
            return path + [state]  
        
        for neighbor, move_cost in next_state_cost(np.array(state)):
            heapq.heappush(pq, (cost + move_cost, neighbor.tolist(), path + [state]))
    end_time = time.time()
    caculation_time =end_time - start_time
    return None  
def dls_iterative(initial, goal, depth_limit):
    global expansion_state
    stack = [(initial.tolist(), [], 0)] 
    visited = set()

    while stack:
        state, path, depth = stack.pop()
        state_tuple = tuple(map(tuple, state))

        if np.array_equal(state, goal):
            return path + [state]

        if depth < depth_limit:
            visited.add(state_tuple)
            expansion_state +=1
            for neighbor in next_state(np.array(state)):
                neighbor_tuple = tuple(map(tuple, neighbor))
                if neighbor_tuple not in visited:
                    stack.append((neighbor, path + [state], depth + 1))

    return None  

def ids(initial, goal):
    global caculation_time,expansion_state
    expansion_state = 0
    start_time = time.time()
    depth = 0
    while True:
        result = dls_iterative(initial, goal, depth)
        if result:
            end_time = time.time()
            caculation_time = end_time -start_time
            return result  
        depth += 1  
def manhattan_distance(state, goal):
    distance = 0
    for num in range(1, 9): 
        x1, y1 = np.where(state == num)
        x2, y2 = np.where(goal == num)
        distance += abs(x1[0] - x2[0]) + abs(y1[0] - y2[0])
    return distance
def gbfs(initial, goal):
    global caculation_time,expansion_state
    expansion_state = 0
    start_time = time.time()
    pq = []
    heapq.heappush(pq, (manhattan_distance(initial, goal), initial.tolist(), []))  
    visited = set()

    while pq:
        _, state, path = heapq.heappop(pq)
        state_tuple = tuple(map(tuple, state))

        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        expansion_state+=1
        if np.array_equal(state, goal):
            end_time = time.time()
            caculation_time = end_time - start_time
            return path + [state]  
        for neighbor in next_state(np.array(state)):
            heapq.heappush(pq, (manhattan_distance(neighbor, goal), neighbor.tolist(), path + [state]))
    end_time = time.time()
    caculation_time = end_time - start_time    
    return None  
def A(initial, goal):
    global caculation_time,expansion_state
    expansion_state = 0
    start_time = time.time()
    pq = []
    heapq.heappush(pq,(0,0,initial.tolist(),[]))
    visited = set()
    while pq:
        _,g, state, path = heapq.heappop(pq)
        state_tuple = tuple(map(tuple, state))

        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        expansion_state +=1
        if np.array_equal(state, goal):
            end_time = time.time()
            caculation_time = end_time - start_time
            return path + [state]  
        for neighbor,cost in next_state_cost(np.array(state)):
            f = g + cost + manhattan_distance(np.array(state),np.array(goal))
            heapq.heappush(pq, (f, g+ cost, neighbor.tolist(), path + [state]))
    
    end_time = time.time()
    caculation_time = end_time - start_time    
    return None
def ida(initial, goal):
    global caculation_time,expansion_state
    expansion_state = 0
    start_time = time.time()
    costlimit = 0
    while True:
        result = A_cost(initial, goal, costlimit)
        if result:
            end_time = time.time()
            caculation_time = end_time - start_time
            return result  
        costlimit += 5 
def A_cost(initial, goal,costlimit):
    global expansion_state
    pq = []
    heapq.heappush(pq,(0,0,initial.tolist(),[]))
    visited = set()
    while pq:
        f,g, state, path = heapq.heappop(pq)
        state_tuple = tuple(map(tuple, state))
        if f > costlimit:
            continue
        if state_tuple in visited:
            continue
        if np.array_equal(state, goal):
            return path + [state]  
        visited.add(state_tuple)
        expansion_state +=1
        if g < costlimit:
            for neighbor,cost in next_state_cost(np.array(state)):
                f = g + cost + manhattan_distance(np.array(state),np.array(goal))
                heapq.heappush(pq, (f, g+ cost, neighbor.tolist(), path + [state]))
    return None
def SHC(initial, goal):
    global caculation_time, expansion_state
    max_iterations = 100000
    expansion_state = 0
    start_time = time.time()
    iteration = 0
    state = initial
    path = [state]
    best_neighbor = None
    best_value = manhattan_distance(state, goal)
    while iteration <= max_iterations:

        
        neighbors = next_state(state) 
        if not neighbors:  
            break
            
        
        neighbor = random.choice(neighbors)
        
        expansion_state += 1
        value = manhattan_distance(neighbor, goal)
        
        if value < best_value:
            best_value = value
            best_neighbor = neighbor

            state = best_neighbor
            path.append(state)
        if np.array_equal(state, goal):
            break
        iteration +=1
    
    end_time = time.time()
    caculation_time = end_time - start_time
    return path if np.array_equal(state, goal) else None
def SAHC(initial, goal):
    global caculation_time, expansion_state
    expansion_state = 0
    start_time= time.time()
    max_iterations = 100
    current_state = initial
    current_score = manhattan_distance(current_state, goal)
    iteration = 0
    path = [current_state]

    while iteration <=max_iterations:
        neighbors = next_state(current_state)
        best_neighbor = None
        best_value = current_score

        for neighbor in neighbors:
            expansion_state +=1
            value = manhattan_distance(neighbor, goal)
            if value < best_value:
                best_value = value
                best_neighbor = neighbor

        if best_neighbor is None or best_value >= current_score:
            break
        
        current_state = best_neighbor
        current_score = best_value
        if np.array_equal(current_state, goal):
            path.append(current_state)
            break
        path.append(current_state)
        iteration += 1
    end_time = time.time()
    caculation_time = end_time - start_time
    return path if np.array_equal(current_state, goal) else None
def stochastic_HC(initial, goal):
    max_iterations=100
    global caculation_time, expansion_state
    expansion_state = 0
    start_time = time.time()
    current_state = initial
    current_score = manhattan_distance(current_state, goal)
    iteration = 0
    path = [current_state]

    while iteration < max_iterations:
        neighbors = next_state(current_state)
        best_value = current_score
        best_neighbors = []

        for neighbor in neighbors:
            value = manhattan_distance(neighbor, goal)
            expansion_state +=1
            if value < best_value:
                best_value = value
                best_neighbors = [neighbor]
            elif value == best_value:
                best_neighbors.append(neighbor)

        if not best_neighbors or best_value >= current_score:
            break

        current_state = random.choice(best_neighbors)
        current_score = best_value
        path.append(current_state)
        if np.array_equal(current_state, goal):
            break
        iteration += 1

    end_time = time.time()
    caculation_time = end_time - start_time

    return path if np.array_equal(current_state, goal) else None
def SA(initial,goal):
    global caculation_time, expansion_state
    start_time = time.time()
    expansion_state = 0
    current_state = initial
    T = 1000
    path = [current_state]
    max_iterations=100
    iteration = 0
    while iteration < max_iterations:
        current_value = manhattan_distance(current_state,goal)
        neighbors = next_state(current_state)
        neighbor = random.choice(neighbors)
        neighbor_value = manhattan_distance(neighbor,goal)
        if neighbor_value >= current_value or random.random() < math.exp(-(current_value-neighbor_value) / T):
            expansion_state +=1
            current_state = neighbor
            current_value = neighbor_value
            path.append(current_state)
        T  *= 0.9
        iteration +=1
        if np.array_equal(current_state, goal):
            break
    end_time = time.time()
    caculation_time = end_time- start_time
    return path if np.array_equal(current_state, goal) else None

    
def BeamS(initial, goal):
    global caculation_time, expansion_state
    start_time = time.time()
    expansion_state = 0
    beam_width = 100
    frontier = [(0, initial, [initial])]
    max_iterations = 1000
    for iteration in range(max_iterations):
        frontier = heapq.nsmallest(beam_width, frontier, key=lambda x: x[0])
        
        next_frontier = []
        
        for _, current_state, path in frontier:
            if np.array_equal(current_state,goal):
                end_time = time.time()
                caculation_time = end_time - start_time
                return path
            
            neighbors = next_state(current_state)
            
            for neighbor in neighbors:
                expansion_state +=1
                new_cost = manhattan_distance(neighbor,goal)
                new_path = path + [neighbor]
                next_frontier.append((new_cost, neighbor, new_path))
        
        frontier = next_frontier 
    end_time = time.time()
    caculation_time = end_time - start_time
    return None  

import random, copy

def genetic(start, goal):
    global caculation_time, expansion_state
    start_time = time.time()
    expansion_state = 0
    population_size=200
    generations=1000
    mutation_rate=0.3
    max_moves=50

    moves = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
    


    def apply_moves_and_get_states(state, path):
        s = copy.deepcopy(state)
        zi, zj = np.where(np.array(s) == 0)
        zi, zj = zi[0],zj[0]
        states = [copy.deepcopy(s)]
        for move in path:
            di, dj = moves[move]
            ni, nj = zi + di, zj + dj
            if 0 <= ni < 3 and 0 <= nj < 3:
                s[zi][zj], s[ni][nj] = s[ni][nj], s[zi][zj]
                zi, zj = ni, nj
                states.append(copy.deepcopy(s)) 
            else:
                continue
        return s,states

    def flatten(s): return [cell for row in s for cell in row]
    def fitness(path):
        final_state, _ = apply_moves_and_get_states(start, path)
        return sum([flatten(final_state)[i] == flatten(goal)[i] for i in range(9)])

    def random_path(): return [random.choice("UDLR") for _ in range(random.randint(5, max_moves))]
    def mutate(path):
        path = path[:]
        global expansion_state
        expansion_state +=1
        if random.random() < 0.5 and len(path) > 1:
            path.pop(random.randint(0, len(path)-1))
        else:
            path.insert(random.randint(0, len(path)), random.choice("UDLR"))
        return path

    def crossover(p1, p2):
        global expansion_state
        min_len = min(len(p1), len(p2))
        if min_len < 2:
            return p1[:]
        cut = random.randint(1, min_len - 1)
        expansion_state += 1
        return p1[:cut] + p2[cut:]

    population = [random_path() for _ in range(population_size)]

    for gen in range(generations):
        population = sorted(population, key=lambda x: -fitness(x))
        best = population[0]
        final_state, states = apply_moves_and_get_states(start, best)
        if np.array_equal( final_state, goal):
            end_time = time.time()
            caculation_time = end_time - start_time
            return states 
        next_gen = population[:10]
        while len(next_gen) < population_size:
            p1, p2 = random.choices(population[:50], k=2)
            child = crossover(p1, p2)
            if random.random() < mutation_rate:
                child = mutate(child)
            next_gen.append(child)
        population = next_gen
    end_time = time.time()
    caculation_time = end_time - start_time
    return None

def move(state, direction):
    i, j = np.where(np.array(state) == 0)
    i , j = i[0],j[0]
    new_state = copy.deepcopy(state)
    if direction == 'up' and i > 0:
        new_state[i][j], new_state[i - 1][j] = new_state[i - 1][j], new_state[i][j]
    elif direction == 'down' and i < 2:
        new_state[i][j], new_state[i + 1][j] = new_state[i + 1][j], new_state[i][j]
    elif direction == 'left' and j > 0:
        new_state[i][j], new_state[i][j - 1] = new_state[i][j - 1], new_state[i][j]
    elif direction == 'right' and j < 2:
        new_state[i][j], new_state[i][j + 1] = new_state[i][j + 1], new_state[i][j]
    else:
        return None
    return new_state

def and_or_search(start, goal):
    global caculation_time, expansion_state
    start_time = time.time()
    expansion_state = 0
    max_depth=50
    def or_search(state, path, visited, depth):
        global expansion_state
        if np.array_equal(state,goal):
            return path + [state]

        if tuple(map(tuple, state))in visited or depth > max_depth:
            return None

        visited.add(tuple(map(tuple, state)))
        expansion_state+=1

        for new_state in next_state(state):
            if tuple(map(tuple, new_state)) not in visited:
                plan = and_search(new_state, path + [state], visited, depth + 1)
                if plan is not None:
                    return plan

        visited.remove(tuple(map(tuple, state))) 
        return None

    def and_search(state, path, visited, depth):
        return or_search(state, path, visited, depth)
    end_time = time.time()
    caculation_time = end_time - start_time
    return or_search(start, [], set(), 0)
def get_initial_state():
    try:
        numbers = [int(cb.get()) for cb in comboboxes]
        if len(set(numbers)) != 9:
            messagebox.showerror("Lỗi", "Các số không được trùng nhau!")
            return None
        return np.array(numbers).reshape(3, 3)
    except ValueError:
        messagebox.showerror("Lỗi", "Vui lòng chọn đầy đủ các số!")
        return None
def get_initial_state1():
    try:
        numbers = [int(cb1.get()) for cb1 in comboboxes1]
        if len(set(numbers)) != 9:
            messagebox.showerror("Lỗi", "Các số không được trùng nhau!")
            return None
        return np.array(numbers).reshape(3, 3)
    except ValueError:
        messagebox.showerror("Lỗi", "Vui lòng chọn đầy đủ các số!")
        return None


solution = []
solution2 = []
present_res = 0
ACTIONS = ['up', 'down', 'left', 'right']
def aftermove(state,path):
    res =[state]
    for i in path :
        if move(state,i) is not None:
            state = move(state,i)
        res.append(state)
    return res
def bfs_two_states(start1,start2 ,goal):
    global caculation_time, expansion_state
    start_time = time.time()
    expansion_state = 0
    visited = set()
    queue = deque()
    queue.append( (start1, start2, []) )

    def serialize(s1, s2):
        return str(s1) + "|" + str(s2)
    while queue:

        s1, s2, path = queue.popleft()
        if np.array_equal(s1,goal) and np.array_equal(s2,goal):
            end_time = time.time()
            caculation_time = end_time -start_time
            return path 
        key = serialize(s1, s2)
        if key in visited:
            continue
        visited.add(key)
        expansion_state+=1
        for action in ACTIONS:
            ns1 = move(s1, action)
            ns2 = move(s2, action)
            if ns1 is not None or ns2 is not None:
                if ns1 is None:
                    ns1 = s1
                if ns2 is None:
                    ns2 =s2
                queue.append((ns1, ns2, path + [action]))
    end_time = time.time()
    caculation_time = end_time -start_time
    return None  
def Partially(start1,start2,goal):
    global caculation_time
    start_time = time.time()
    def bfsp(initial, goal):
        global expansion_state
        expansion_state = 0
        queue = deque([(initial.tolist(), [])])
        visited = set()
        
        while queue:
            state, path = queue.popleft()
            state_tuple = tuple(map(tuple, state))
            
            if state_tuple in visited:
                continue
            visited.add(state_tuple)
            expansion_state +=1
            if np.array_equal(state, goal):
                end_time = time.time()
                caculation_time =end_time - start_time
                return path + [state] 
            
            for neighbor in next_state(np.array(state)):
                if neighbor[0][0] == 1:
                    queue.append((neighbor.tolist(), path + [state]))
                else:
                    continue

        return None
    end_time = time.time()
    caculation_time =end_time - start_time
    return bfsp(start1,goal), bfsp(start2,goal)
def KiemThu():
    global caculation_time, expansion_state
    start_time = time.time()
    expansion_state = 0
    res = []
    numbers = list(range(9))
    for p in itertools.permutations(numbers):
        expansion_state +=1
        grid = [list(p[i:i+3]) for i in range(0, 9, 3)]
        valid = True
        for col in range(3): 
            for row in range(2):
                top = grid[row][col]    
                bottom = grid[row + 1][col] 
                if bottom != top + 3: 
                    valid = False
                    break
            if not valid:
                break
        if valid:
            for row in range(3): 
                for col in range(2):
                    left = grid[row][col]    
                    right = grid[row][col+1] 
                    if left != right -1: 
                        valid = False
                        break
                if not valid:
                    break        

        if valid:
            res.append(grid)
            end_time = time.time()
            caculation_time = end_time - start_time
    return res 

def is_valid(value, assignment):
    return value not in assignment

def backtrack(assignment=None):
    global expansion_state
    expansion_state = 0
    
    if assignment is None:
        assignment = []

    if len(assignment) == 9:
        return assignment
    
    for value in range(9):
        if is_valid(value, assignment):
            assignment.append(value)
            result = backtrack(assignment)
            if result is not None:
                result = np.array(result).reshape((3, 3)).tolist()
                return [result]
            assignment.pop()
    return None

def ac3():
    global expansion_state
    expansion_state = 0
    variables = [f"X{i}" for i in range(1, 10)]
    domains = {var: set(range(9)) for var in variables}

    def rangbuoc(xi, xj, x, y):
        i = int(xi[1:])
        j = int(xj[1:])
        if abs(i - j) == 1:
            if i + 1 == j:
                return x + 1 == y
            elif j + 1 == i:
                return x == y + 1
        return x != y

    def arcs():
        arcs = []
        for i in range(len(variables)):
            for j in range(len(variables)):
                if i != j:
                    arcs.append((variables[i], variables[j]))
        return deque(arcs)

    def rm_inconsistent_values(domains, xi, xj):
        removed = False
        to_remove = set()
        for x in domains[xi]:
            if not any(rangbuoc(xi, xj, x, y) for y in domains[xj]):
                to_remove.add(x)
        if to_remove:
            domains[xi] -= to_remove
            removed = True
        return removed

    def ac3(domains):

        queue = arcs()
        while queue:
            xi, xj = queue.popleft()
            if rm_inconsistent_values(domains, xi, xj):
                for xk in variables:
                    if xk != xi and xk != xj:
                        queue.append((xk, xi))
        return domains

    reduced_domains = ac3(deepcopy(domains))
    solution = [list(reduced_domains[f"X{i}"])[0] for i in range(1, 10)]
    solution = [solution[i:i+3] for i in range(0, 9, 3)]
    return [solution]

def Q_Learing(initial, goal):
    global caculation_time, expansion_state
    start_time = time.time()
    expansion_state = 0
    alpha = 0.1     
    gamma = 0.9     
    epsilon = 0.2   
    episodes = 1000
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def possibles(state):
        i, j = np.where(np.array(state) == 0)
        i , j = i[0],j[0]
        possible = []
        if i > 0: possible.append('up')
        if i < 2: possible.append('down')
        if j > 0: possible.append('left')
        if j < 2: possible.append('right')
        return possible
    def reward_f(next_state):
        return 100 if np.array_equal(np.array(next_state),np.array(goal))  else -1
    def choose_action(state):
        if random.random() < epsilon:
            return random.choice(possibles(state))
        else:
            q_vals = Q[tuple(map(tuple, state))]
            actions_valid = possibles(state)
            return max(actions_valid, key=lambda a: q_vals[a])
    def train(start_state):
        global expansion_state
        for ep in range(episodes):
            state = start_state
            steps = 0
            while not np.array_equal(np.array(state),np.array(goal)) and steps < 100:
                action = choose_action(state)
                next_state = move(state, action)
                reward = reward_f(next_state)
                expansion_state+=1
                best_next = max(Q[tuple(map(tuple, next_state))].values())
                Q[tuple(map(tuple, state))][action] += alpha * (reward + gamma * best_next - Q[tuple(map(tuple, state))][action])
                state = next_state
                steps += 1
    def solve(start_state):
        state = start_state
        path = [state]
        steps = 0
        while not np.array_equal(np.array(state),np.array(goal))  and steps < 100:
            action = choose_action(state)
            state = move(state, action)
            path.append(state)
            steps += 1
        return path
    train(initial)
    end_time = time.time()
    caculation_time = end_time - start_time
    return solve(initial)

def update_display():
    if solution and 0 <= present_res < len(solution):
        state = solution[present_res]
        for i in range(3):
            for j in range(3):
                value = state[i][j]
                labels[i][j].config(text=str(value) if value != 0 else " ")
        Displaystep.set(f"Bước {present_res} / { max(len(solution),-1 if solution2 == None else len (solution2)) - 1}")
    
    
    if solution2 and 0 <= present_res < len(solution2):
        state = solution2[present_res]
        for i in range(3):
            for j in range(3):
                value = state[i][j]
                labels1[i][j].config(text=str(value) if value != 0 else " ")
        #Displaystep.set(f"Bước {present_res} / {len(solution2)-1}")
    

def previous_step():
    global present_res
    if present_res > 0:
        present_res -= 1
        update_display()

def next_step():
    global present_res
    if present_res <  max(len(solution),-1 if solution2 == None else len (solution2)) - 1:
        present_res += 1
        update_display()
auto_running = False
def auto_move():
    global present_res,solution,auto_running
    if auto_running and  present_res <  max(len(solution),-1 if solution2 == None else len (solution2)) - 1:
        present_res += 1
        update_display()
        root.after(50, auto_move)
def toggle_auto():
    global auto_running
    auto_running = not auto_running
    auto_button.config(text="Dừng" if auto_running else "Tự động") 
    if auto_running:
        auto_move()

root = tk.Tk()
root.title("8 Puzzle - TRỊNH QUỐC CÔNG VINH")
root.geometry("800x1000")
root.resizable(False, False)

values = list(range(0, 9))
comboboxes = []
comboboxes1 = []
tk.Label(root, text="Nhập trạng thái ban đầu:", font=("Arial", 12, "bold")).pack(pady=10)

frame = tk.Frame(root)
frame.pack(side="top")
for i in range(3):
    for j in range(3):
        cb = ttk.Combobox(frame, values=values, state="readonly", width=3, font=("Arial", 14))
        cb.grid(row=i, column=j, padx=5, pady=5)
        comboboxes.append(cb)
        cb.current(example_state[i][j]) 
for i in range(3):
    for j in range(3):
        cb1 = ttk.Combobox(frame, values=values, state="readonly", width=3, font=("Arial", 14))
        cb1.grid(row=i, column=j + 4, padx=5, pady=5)
        comboboxes1.append(cb1)
        cb1.current(example_state2[i][j]) 
b =tk.Button(frame,state="disabled",relief="flat", bg=root.cget('bg'))
b.grid(column=3,row=0)

frame.grid_columnconfigure(3, weight=1, pad=50)


solveframe = tk.Frame(root)
solveframe.pack()
bfs_button = tk.Button(solveframe, text="BFS", font=("Arial", 12, "bold"), command=solve_command(bfs))
bfs_button.grid(row=0,column=0,padx= 5,pady=5)

dfs_button =  tk.Button(solveframe, text="DFS", font=("Arial", 12, "bold"), command=solve_command(dfs))
dfs_button.grid(row=0,column=1,padx= 5,pady=5)

ucs_button =  tk.Button(solveframe, text="UCS", font=("Arial", 12, "bold"), command=solve_command(ucs))
ucs_button.grid(row=0,column=2,padx= 5,pady=5)

ids_button =  tk.Button(solveframe, text="IDS", font=("Arial", 12, "bold"), command=solve_command(ids))
ids_button.grid(row=0,column=3,padx= 5,pady=5)


gbfs_button =  tk.Button(solveframe, text="GBFS", font=("Arial", 12, "bold"), command=solve_command(gbfs))
gbfs_button.grid(row=0,column=4,padx= 5,pady=5)

A_button =  tk.Button(solveframe, text="A*", font=("Arial", 12, "bold"), command=solve_command(A))
A_button.grid(row=0,column=5,padx= 5,pady=5)

A_costbutton =  tk.Button(solveframe, text="IDA*", font=("Arial", 12, "bold"), command=solve_command(ida))
A_costbutton.grid(row=1,column=0,padx= 5,pady=5)


SHC_button =  tk.Button(solveframe, text="SHC", font=("Arial", 12, "bold"), command=solve_command(SHC))
SHC_button.grid(row=1,column=1,padx= 5,pady=5)

SAHC_button =  tk.Button(solveframe, text="SAHC", font=("Arial", 12, "bold"), command=solve_command(SAHC))
SAHC_button.grid(row=1,column=2,padx= 5,pady=5)


StHC_button =  tk.Button(solveframe, text="sto_HC", font=("Arial", 12, "bold"), command=solve_command(stochastic_HC))
StHC_button.grid(row=1,column=3,padx= 5,pady=5)

SA_button =  tk.Button(solveframe, text="SA", font=("Arial", 12, "bold"), command=solve_command(SA))
SA_button.grid(row=1,column=4,padx= 5,pady=5)

BeamS_button =  tk.Button(solveframe, text="Beam", font=("Arial", 12, "bold"), command=solve_command(BeamS))
BeamS_button.grid(row=2,column=0,padx= 5,pady=5)

Gen_button =  tk.Button(solveframe, text="Gen", font=("Arial", 12, "bold"), command=solve_command(genetic))
Gen_button.grid(row=2,column=1,padx= 5,pady=5)

AO_button =  tk.Button(solveframe, text="AO", font=("Arial", 12, "bold"), command=solve_command(and_or_search))
AO_button.grid(row=2,column=2,padx= 5,pady=5)


belief_state_button =  tk.Button(solveframe, text="Belief", font=("Arial", 12, "bold"), command=solve_command(bfs_two_states))
belief_state_button.grid(row=2,column=3,padx= 5,pady=5)



Partially_button =  tk.Button(solveframe, text="Partially", font=("Arial", 12, "bold"), command=solve_command(Partially))
Partially_button.grid(row=2,column=4,padx= 5,pady=5)

KiemThu_button =  tk.Button(solveframe, text="KiemThu", font=("Arial", 12, "bold"), command=solve_command(KiemThu))
KiemThu_button.grid(row=2,column=5,padx= 5,pady=5)

backtrack_button =  tk.Button(solveframe, text="Backtrack", font=("Arial", 12, "bold"), command=solve_command(backtrack))
backtrack_button.grid(row=3,column=0,padx= 5,pady=5)

ac3_button =  tk.Button(solveframe, text="ac3", font=("Arial", 12, "bold"), command=solve_command(ac3))
ac3_button.grid(row=3,column=1,padx= 5,pady=5)


Q_button =  tk.Button(solveframe, text="Q-L", font=("Arial", 12, "bold"), command=solve_command(Q_Learing))
Q_button.grid(row=3,column=2,padx= 5,pady=5)

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Arial", 12, "bold"), fg="blue")
result_label.pack()

frame_res = tk.Frame(root)
frame_res.pack()
labels = [[tk.Label(frame_res, text="", font=("Arial", 16), width=4, height=2, relief="solid") 
           for _ in range(3)] for _ in range(3)]
for i in range(3):
    for j in range(3):
        labels[i][j].grid(row=i, column=j, padx=5, pady=5)
labels1 = [[tk.Label(frame_res, text="", font=("Arial", 16), width=4, height=2, relief="solid") 
           for _ in range(3)] for _ in range(3)]
for i in range(3):
    for j in range(3):
        labels1[i][j].grid(row=i, column=j+4, padx=5, pady=5)

c =tk.Button(frame_res,state="disabled",relief="flat", bg=root.cget('bg'))
c.grid(column=3,row=0)

frame_res.grid_columnconfigure(3, weight=1, pad=100)
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

prev_button = tk.Button(button_frame, text="\u2190 Trước", font=("Arial", 12, "bold"), command=previous_step)
prev_button.pack(side=tk.LEFT, padx=10)
auto_button = tk.Button(button_frame, text="Tự động", font=("Arial", 12, "bold"), command=toggle_auto)
auto_button.pack(side=tk.LEFT, padx=10)

next_button = tk.Button(button_frame, text="Sau \u2192", font=("Arial", 12, "bold"), command=next_step)
next_button.pack(side=tk.RIGHT, padx=10)
Displaystep = tk.StringVar()

labelstep = tk.Label(root,textvariable=Displaystep,font=("Arial", 12, "bold"))
labelstep.pack()
goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])

root.mainloop()

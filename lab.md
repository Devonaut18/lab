**1\. A\* Search for 8-Puzzle Problem**

```py
import heapq

class Node:
    def __init__(self, state, parent, move, g, h):
        self.state = state      # Current board configuration
        self.parent = parent    # Parent node to trace path
        self.move = move        # Move taken to get here (U, D, L, R)
        self.g = g              # Cost from start
        self.h = h              # Heuristic cost to goal
        self.f = g + h          # Total cost

    # Required for heapq to compare nodes based on f-score
    def __lt__(self, other):
        return self.f < other.f

# Heuristic: Manhattan Distance
def get_manhattan_distance(state, goal):
    distance = 0
    for i in range(1, 9): # Tiles 1 through 8 (ignore 0)
        curr_idx = state.index(i)
        goal_idx = goal.index(i)
        
        # Convert 1D index to 2D (x, y) coordinates
        curr_x, curr_y = curr_idx % 3, curr_idx // 3
        goal_x, goal_y = goal_idx % 3, goal_idx // 3
        
        distance += abs(curr_x - goal_x) + abs(curr_y - goal_y)
    return distance

# Generate valid adjacent states
def get_neighbors(state):
    neighbors = []
    zero_idx = state.index(0)
    x, y = zero_idx % 3, zero_idx // 3
    
    moves = [('Up', 0, -1), ('Down', 0, 1), ('Left', -1, 0), ('Right', 1, 0)]
    
    for move, dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_idx = ny * 3 + nx
            new_state = list(state)
            new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
            neighbors.append((tuple(new_state), move))
            
    return neighbors

def a_star(start_state, goal_state):
    open_list = []
    closed_set = set()
    
    h = get_manhattan_distance(start_state, goal_state)
    start_node = Node(start_state, None, None, 0, h)
    heapq.heappush(open_list, start_node)
    
    while open_list:
        current = heapq.heappop(open_list)
        
        if current.state == goal_state:
            path = []
            while current.parent:
                path.append(current.move)
                current = current.parent
            return path[::-1]
            
        closed_set.add(current.state)
        
        for next_state, move in get_neighbors(current.state):
            if next_state in closed_set:
                continue
                
            g_cost = current.g + 1
            h_cost = get_manhattan_distance(next_state, goal_state)
            neighbor_node = Node(next_state, current, move, g_cost, h_cost)
            heapq.heappush(open_list, neighbor_node)
            
    return None

if __name__ == "__main__":
    start = (1, 2, 3, 4, 0, 5, 7, 8, 6)
    goal =  (1, 2, 3, 4, 5, 6, 7, 8, 0)

    print("Solving A*...")
    solution = a_star(start, goal)
    if solution:
        print(f"Solved in {len(solution)} moves! Steps:", solution)
    else:
        print("No solution exists.")
```

---

**2\. Tower of Hanoi by State Space Search (BFS)**

```py
from collections import deque

def get_neighbors(state):
    neighbors = []
    for i in range(3):
        for j in range(3):
            if i != j and state[i]: 
                top_disk = state[i][-1]
                if not state[j] or state[j][-1] > top_disk:
                    new_state = list(state)
                    new_state[i] = state[i][:-1]
                    new_state[j] = state[j] + (top_disk,)
                    move_name = f"Move disk {top_disk} from Peg {i} to Peg {j}"
                    neighbors.append((tuple(new_state), move_name))
    return neighbors

def solve_hanoi(num_disks):
    start_peg = tuple(range(num_disks, 0, -1))
    start_state = (start_peg, (), ())
    goal_state = ((), (), start_peg)
    
    queue = deque([(start_state, [])])
    visited = set([start_state])
    
    while queue:
        current_state, path = queue.popleft()
        if current_state == goal_state:
            return path
            
        for next_state, move in get_neighbors(current_state):
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [move]))
                
    return None

if __name__ == "__main__":
    disks = 3
    print(f"Solving Tower of Hanoi for {disks} disks using BFS...\n")
    solution = solve_hanoi(disks)
    if solution:
        print(f"Solved optimally in {len(solution)} moves:")
        for step_num, step in enumerate(solution, 1):
            print(f"Step {step_num}: {step}")
```

---

**3\. Local Beam Search for Traveling Salesman Problem (TSP)**

```py
import random

DISTANCES = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

def calculate_cost(tour):
    cost = 0
    for i in range(len(tour)):
        cost += DISTANCES[tour[i - 1]][tour[i]]
    return cost

def get_neighbors(tour):
    neighbors = []
    n = len(tour)
    for i in range(n):
        for j in range(i + 1, n):
            neighbor = list(tour)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
    return neighbors

def local_beam_search(cities, k, max_iters):
    beam = [random.sample(cities, len(cities)) for _ in range(k)]
    best_tour = None
    best_cost = float('inf')

    for _ in range(max_iters):
        all_neighbors = []
        for state in beam:
            all_neighbors.extend(get_neighbors(state))
            
        all_neighbors.sort(key=calculate_cost)
        beam = all_neighbors[:k]
        
        current_best = beam[0]
        current_best_cost = calculate_cost(current_best)
        
        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_tour = current_best
            
    return best_tour, best_cost

if __name__ == "__main__":
    cities_list = [0, 1, 2, 3]
    beam_width = 2
    iterations = 50

    best_route, minimum_cost = local_beam_search(cities_list, beam_width, iterations)
    route_display = best_route + [best_route[0]]
    print("Optimal Route Found:", " -> ".join(map(str, route_display)))
    print("Total Distance (Cost):", minimum_cost)
```

---

4\. Breadth-First Search (BFS) for 8-Puzzle

```py
from collections import deque

def get_neighbors(state):
    neighbors = []
    zero_idx = state.index(0)
    x, y = zero_idx % 3, zero_idx // 3
    moves = [('Up', 0, -1), ('Down', 0, 1), ('Left', -1, 0), ('Right', 1, 0)]
    
    for move, dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_idx = ny * 3 + nx
            new_state = list(state)
            new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
            neighbors.append((tuple(new_state), move))
    return neighbors

def bfs_8_puzzle(start_state, goal_state):
    queue = deque([(start_state, [])])
    visited = set([start_state])
    
    while queue:
        current_state, path = queue.popleft()
        if current_state == goal_state:
            return path
            
        for next_state, move in get_neighbors(current_state):
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [move]))
    return None

if __name__ == "__main__":
    start = (1, 2, 3, 4, 0, 6, 7, 5, 8)
    goal =  (1, 2, 3, 4, 5, 6, 7, 8, 0)

    solution = bfs_8_puzzle(start, goal)
    if solution:
        print(f"Solved optimally in {len(solution)} moves! Steps:", solution)
```

---

**5\. Depth-First Search (DFS) for Water Jug Problem**

```py
def dfs_water_jug(capacity1, capacity2, target):
    start_state = (0, 0)
    stack = [(start_state, [])]
    visited = set()
    
    while stack:
        (jug1, jug2), path = stack.pop()
        
        if jug1 == target or jug2 == target:
            path.append((jug1, jug2))
            return path
            
        if (jug1, jug2) in visited:
            continue
        visited.add((jug1, jug2))
        current_path = path + [(jug1, jug2)]
        
        stack.append(((capacity1, jug2), current_path)) # 1. Fill 1
        stack.append(((jug1, capacity2), current_path)) # 2. Fill 2
        stack.append(((0, jug2), current_path))         # 3. Empty 1
        stack.append(((jug1, 0), current_path))         # 4. Empty 2
        
        pour_to_2 = min(jug1, capacity2 - jug2)
        stack.append(((jug1 - pour_to_2, jug2 + pour_to_2), current_path)) # 5. Pour 1->2
        
        pour_to_1 = min(jug2, capacity1 - jug1)
        stack.append(((jug1 + pour_to_1, jug2 - pour_to_1), current_path)) # 6. Pour 2->1
        
    return None

if __name__ == "__main__":
    solution = dfs_water_jug(4, 3, 2)
    if solution:
        print(f"Solution found in {len(solution) - 1} steps:")
        for step_num, state in enumerate(solution):
            print(f"Step {step_num}: {state}")
```

---

**6\. Backtracking Search for 8-Queens**

```py
def is_safe(board, row, col):
    for i in range(col):
        prev_row = board[i]
        if prev_row == row:
            return False
        if abs(prev_row - row) == abs(i - col):
            return False
    return True

def solve_n_queens(board, col, n):
    if col == n:
        return True
        
    for row in range(n):
        if is_safe(board, row, col):
            board[col] = row
            if solve_n_queens(board, col + 1, n):
                return True
            board[col] = -1 
    return False

def print_board(board, n):
    for row in range(n):
        print(" ".join(["Q" if board[col] == row else "." for col in range(n)]))
    print("\nBoard Array Representation:", board)

if __name__ == "__main__":
    N = 8
    chess_board = [-1] * N
    if solve_n_queens(chess_board, 0, N):
        print("Solution found!\n")
        print_board(chess_board, N)
```

---

**7\. Rule-Based Chatbot**

```py
import re
import random

RULES = [
    (r'\b(hi|hello|hey)\b', ["Hello! How can I help you?", "Hi there!"]),
    (r'what is your name', ["I am a simple rule-based chatbot."]),
    (r'how are you', ["Functioning perfectly, thanks!"]),
    (r'\b(bye|exit|quit)\b', ["Goodbye! Have a great day."]),
    (r'(.*)', ["I'm sorry, I didn't quite catch that.", "Could you rephrase that?"]) 
]

def get_response(user_input):
    user_input = user_input.lower()
    for pattern, responses in RULES:
        if re.search(pattern, user_input):
            return random.choice(responses)

if __name__ == "__main__":
    print("Chatbot initialized. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        response = get_response(user_input)
        print(f"Bot: {response}")
        if re.search(r'\b(bye|exit|quit)\b', user_input.lower()):
            break
```

---

8\. Simple Reflex Agent (Vacuum Cleaner World)

```py
def simple_reflex_agent(location, status):
    if status == 'Dirty':
        return 'Suck'
    elif location == 'A':
        return 'Right'
    elif location == 'B':
        return 'Left'

def simulate_vacuum_world(starting_location, room_a_status, room_b_status):
    environment = {'A': room_a_status, 'B': room_b_status}
    current_location = starting_location
    score = 0
    
    print(f"Initial State: Room A is {environment['A']}, Room B is {environment['B']}")
    
    for step in range(1, 4):
        current_status = environment[current_location]
        action = simple_reflex_agent(current_location, current_status)
        
        if action == 'Suck':
            environment[current_location] = 'Clean'
            score += 10
            print(f"Step {step}: Sucked dirt in Room {current_location}.")
        elif action == 'Right':
            current_location = 'B'
            score -= 1
            print(f"Step {step}: Moved Right to B.")
        elif action == 'Left':
            current_location = 'A'
            score -= 1
            print(f"Step {step}: Moved Left to A.")
            
        if environment['A'] == 'Clean' and environment['B'] == 'Clean':
            print("\nGoal Achieved: Both rooms are clean!")
            break
            
    print(f"Final Score: {score}")

if __name__ == "__main__":
    simulate_vacuum_world('A', 'Dirty', 'Dirty')
```

&nbsp;
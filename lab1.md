1\. A\* Search for 8-Puzzle Problem

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
    
    # Directions: (Move Name, dx, dy)
    moves = [('Up', 0, -1), ('Down', 0, 1), ('Left', -1, 0), ('Right', 1, 0)]
    
    for move, dx, dy in moves:
        nx, ny = x + dx, y + dy
        
        # Check if the new position is within the 3x3 grid
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_idx = ny * 3 + nx
            new_state = list(state)
            
            # Swap empty space (0) with the adjacent tile
            new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
            neighbors.append((tuple(new_state), move))
            
    return neighbors

def a_star(start_state, goal_state):
    open_list = []
    closed_set = set()
    
    # Initialize start node
    h = get_manhattan_distance(start_state, goal_state)
    start_node = Node(start_state, None, None, 0, h)
    heapq.heappush(open_list, start_node)
    
    while open_list:
        current = heapq.heappop(open_list)
        
        # Goal Check
        if current.state == goal_state:
            path = []
            while current.parent:
                path.append(current.move)
                current = current.parent
            return path[::-1] # Return reversed path
            
        closed_set.add(current.state)
        
        # Explore neighbors
        for next_state, move in get_neighbors(current.state):
            if next_state in closed_set:
                continue
                
            g_cost = current.g + 1
            h_cost = get_manhattan_distance(next_state, goal_state)
            neighbor_node = Node(next_state, current, move, g_cost, h_cost)
            
            heapq.heappush(open_list, neighbor_node)
            
    return None # No solution found

# --- Testing the code ---
if __name__ == "__main__":
    # 0 represents the empty space
    start = (1, 2, 3, 
             4, 0, 5, 
             7, 8, 6)
             
    goal =  (1, 2, 3, 
             4, 5, 6, 
             7, 8, 0)

    print("Solving A*...")
    solution = a_star(start, goal)
    
    if solution:
        print(f"Solved in {len(solution)} moves!")
        print("Steps:", solution)
    else:
        print("No solution exists.")
```

**2\. Tower of Hanoi by State Space Search (BFS)**

```py
from collections import deque

# Generate valid next board configurations
def get_neighbors(state):
    neighbors = []
    
    # Check every possible peg pair (moving from peg 'i' to peg 'j')
    for i in range(3):
        for j in range(3):
            if i != j and state[i]: # If moving to a different peg and source peg is not empty
                top_disk = state[i][-1]
                
                # Rule: Destination peg must be empty OR its top disk must be larger
                if not state[j] or state[j][-1] > top_disk:
                    
                    # Create a mutable copy of the state
                    new_state = list(state)
                    
                    # Remove disk from source peg (i) and add to destination peg (j)
                    new_state[i] = state[i][:-1]
                    new_state[j] = state[j] + (top_disk,)
                    
                    # Store (new configuration, move representation)
                    move_name = f"Move disk {top_disk} from Peg {i} to Peg {j}"
                    neighbors.append((tuple(new_state), move_name))
                    
    return neighbors

def solve_hanoi(num_disks):
    # Setup Start and Goal States
    # For 3 disks, start_peg = (3, 2, 1)
    start_peg = tuple(range(num_disks, 0, -1))
    start_state = (start_peg, (), ())
    goal_state = ((), (), start_peg)
    
    # Queue stores: (current_state, path_of_moves)
    queue = deque([(start_state, [])])
    visited = set([start_state])
    
    while queue:
        current_state, path = queue.popleft()
        
        # Goal Check
        if current_state == goal_state:
            return path
            
        # Explore valid moves
        for next_state, move in get_neighbors(current_state):
            if next_state not in visited:
                visited.add(next_state)
                # Add new state to queue along with the updated path history
                queue.append((next_state, path + [move]))
                
    return None # Failsafe, though a solution always exists

# --- Testing the code ---
if __name__ == "__main__":
    disks = 3
    print(f"Solving Tower of Hanoi for {disks} disks using BFS...\n")
    
    solution = solve_hanoi(disks)
    
    if solution:
        print(f"Solved optimally in {len(solution)} moves:")
        for step_num, step in enumerate(solution, 1):
            print(f"Step {step_num}: {step}")
    else:
        print("No solution found.")
```

**3\. Local Beam Search for Traveling Salesman Problem (TSP)**

```py
import random

# Example distance matrix for 4 cities (0, 1, 2, 3)
# DISTANCES[i][j] is the distance from city i to city j
DISTANCES = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

# Calculate total distance of a given tour
def calculate_cost(tour):
    cost = 0
    for i in range(len(tour)):
        # tour[i - 1] to tour[i] automatically handles the return to start 
        # because when i=0, tour[-1] is the last city in the list.
        cost += DISTANCES[tour[i - 1]][tour[i]]
    return cost

# Generate all possible neighbors by swapping two cities
def get_neighbors(tour):
    neighbors = []
    n = len(tour)
    for i in range(n):
        for j in range(i + 1, n):
            # Create a copy and swap elements at index i and j
            neighbor = list(tour)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
    return neighbors

def local_beam_search(cities, k, max_iters):
    # Step 1: Initialize k random start states
    beam = [random.sample(cities, len(cities)) for _ in range(k)]
    
    best_tour = None
    best_cost = float('inf')

    # Step 2: Loop for a fixed number of iterations
    for _ in range(max_iters):
        all_neighbors = []
        
        # Generate all successors for all k states in the beam
        for state in beam:
            all_neighbors.extend(get_neighbors(state))
            
        # Step 3: Sort all generated neighbors by their cost (ascending)
        all_neighbors.sort(key=calculate_cost)
        
        # Step 4: Keep the best k neighbors for the next iteration
        beam = all_neighbors[:k]
        
        # Track the absolute best route found so far
        current_best = beam[0]
        current_best_cost = calculate_cost(current_best)
        
        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_tour = current_best
            
    return best_tour, best_cost

# --- Testing the code ---
if __name__ == "__main__":
    cities_list = [0, 1, 2, 3]
    beam_width = 2
    iterations = 50

    print(f"Solving TSP using Local Beam Search (k={beam_width})...")
    best_route, minimum_cost = local_beam_search(cities_list, beam_width, iterations)
    
    # Format the output to show the closed loop
    route_display = best_route + [best_route[0]]
    
    print("\nOptimal Route Found:", " -> ".join(map(str, route_display)))
    print("Total Distance (Cost):", minimum_cost)
```

**4\. Breadth-First Search (BFS) for 8-Puzzle**

```py
from collections import deque

# Generate valid adjacent states
def get_neighbors(state):
    neighbors = []
    # Find the index of the empty space (0)
    zero_idx = state.index(0)
    x, y = zero_idx % 3, zero_idx // 3
    
    # Directions: (Move Name, dx, dy)
    moves = [('Up', 0, -1), ('Down', 0, 1), ('Left', -1, 0), ('Right', 1, 0)]
    
    for move, dx, dy in moves:
        nx, ny = x + dx, y + dy
        
        # Check if the new position is within the 3x3 grid boundaries
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_idx = ny * 3 + nx
            new_state = list(state)
            
            # Swap empty space (0) with the adjacent tile
            new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
            
            neighbors.append((tuple(new_state), move))
            
    return neighbors

def bfs_8_puzzle(start_state, goal_state):
    # Queue stores tuples of: (current_board_state, path_of_moves_taken)
    queue = deque([(start_state, [])])
    
    # Set to keep track of visited configurations to prevent loops
    visited = set([start_state])
    
    while queue:
        current_state, path = queue.popleft()
        
        # Goal Check
        if current_state == goal_state:
            return path
            
        # Explore valid neighbors
        for next_state, move in get_neighbors(current_state):
            if next_state not in visited:
                visited.add(next_state)
                # Enqueue the neighbor with the new move appended to the path history
                queue.append((next_state, path + [move]))
                
    return None # No solution exists

# --- Testing the code ---
if __name__ == "__main__":
    # 0 represents the empty space
    start = (1, 2, 3, 
             4, 0, 6, 
             7, 5, 8)
             
    goal =  (1, 2, 3, 
             4, 5, 6, 
             7, 8, 0)

    print("Solving 8-Puzzle using BFS...")
    solution = bfs_8_puzzle(start, goal)
    
    if solution:
        print(f"Solved optimally in {len(solution)} moves!")
        print("Steps:", solution)
    else:
        print("No solution exists.")
```

**5\. Depth-First Search (DFS) for Water Jug Problem**

```py
def dfs_water_jug(capacity1, capacity2, target):
    # Stack stores tuples of: (current_state, path_of_states_taken)
    start_state = (0, 0)
    stack = [(start_state, [])]
    
    # Set to prevent infinite loops
    visited = set()
    
    while stack:
        # LIFO: Pop from the end of the list
        (jug1, jug2), path = stack.pop()
        
        # Goal Check: Does either jug hold the target amount?
        if jug1 == target or jug2 == target:
            path.append((jug1, jug2)) # Add the final winning state to the path
            return path
            
        # Cycle checking
        if (jug1, jug2) in visited:
            continue
            
        visited.add((jug1, jug2))
        
        # Record the current state in our path history
        current_path = path + [(jug1, jug2)]
        
        # --- Generate neighbors using the 6 rules ---
        
        # 1. Fill Jug 1
        stack.append(((capacity1, jug2), current_path))
        
        # 2. Fill Jug 2
        stack.append(((jug1, capacity2), current_path))
        
        # 3. Empty Jug 1
        stack.append(((0, jug2), current_path))
        
        # 4. Empty Jug 2
        stack.append(((jug1, 0), current_path))
        
        # 5. Pour Jug 1 -> Jug 2
        # Amount poured is limited by either Jug 1's content or Jug 2's remaining space
        pour_to_2 = min(jug1, capacity2 - jug2)
        stack.append(((jug1 - pour_to_2, jug2 + pour_to_2), current_path))
        
        # 6. Pour Jug 2 -> Jug 1
        # Amount poured is limited by either Jug 2's content or Jug 1's remaining space
        pour_to_1 = min(jug2, capacity1 - jug1)
        stack.append(((jug1 + pour_to_1, jug2 - pour_to_1), current_path))
        
    return None # No solution exists

# --- Testing the code ---
if __name__ == "__main__":
    jug1_cap = 4
    jug2_cap = 3
    target_amount = 2

    print(f"Solving Water Jug ({jug1_cap}G, {jug2_cap}G) for Target {target_amount}G using DFS...\n")
    solution = dfs_water_jug(jug1_cap, jug2_cap, target_amount)
    
    if solution:
        print(f"Solution found in {len(solution) - 1} steps:")
        print("(Jug 1, Jug 2)")
        for step_num, state in enumerate(solution):
            print(f"Step {step_num}: {state}")
    else:
        print("No solution exists.")
```

**6\. Backtracking Search for 8-Queens**

```py
# Check if a queen can be safely placed at (row, col)
def is_safe(board, row, col):
    # Check against all previously placed queens (columns 0 to col-1)
    for i in range(col):
        prev_row = board[i]
        
        # Check horizontal attack (same row)
        if prev_row == row:
            return False
            
        # Check diagonal attack
        if abs(prev_row - row) == abs(i - col):
            return False
            
    return True

# The Backtracking function
def solve_n_queens(board, col, n):
    # Base case: If all queens are placed, we found a solution
    if col == n:
        return True
        
    # Try placing a queen in all rows of the current column
    for row in range(n):
        if is_safe(board, row, col):
            # Place the queen
            board[col] = row
            
            # Recurse to place the rest of the queens
            if solve_n_queens(board, col + 1, n):
                return True
                
            # BACKTRACK: Remove the queen and try the next row
            # (Implicitly done by overwriting in the next loop, 
            # but explicitly setting to -1 makes the concept clear)
            board[col] = -1 
            
    # If no row works in this column, return False to trigger backtracking
    return False

# Helper function to print the board nicely
def print_board(board, n):
    for row in range(n):
        row_str = ""
        for col in range(n):
            if board[col] == row:
                row_str += " Q "
            else:
                row_str += " . "
        print(row_str)
    print("\nBoard Array Representation:", board)

# --- Testing the code ---
if __name__ == "__main__":
    N = 8
    # board[i] will store the row index of the queen in column i
    # Initialize with -1 (no queen placed)
    chess_board = [-1] * N
    
    print(f"Solving {N}-Queens using Backtracking...\n")
    
    if solve_n_queens(chess_board, 0, N):
        print("Solution found!\n")
        print_board(chess_board, N)
    else:
        print("No solution exists.")
```

**7\. Rule-Based Chatbot**

```py
import re
import random

# Define rules as a list of tuples: (Regex Pattern, List of Responses)
# The bot will check these in order from top to bottom.
RULES = [
    (r'\b(hi|hello|hey|greetings)\b', 
     ["Hello! How can I help you?", "Hi there!", "Greetings human!"]),
     
    (r'what is your name', 
     ["I am a simple rule-based chatbot.", "They call me PyBot."]),
     
    (r'how are you', 
     ["I'm just a bunch of code, but I'm doing great!", "Functioning perfectly, thanks!"]),
     
    (r'tell me a joke', 
     ["Why did the computer keep freezing? Because it left its Windows open!", 
      "There are 10 types of people: those who understand binary, and those who don't."]),
      
    (r'my name is (.*)', 
     ["Nice to meet you!", "Hello! That's a great name."]),
     
    (r'\b(bye|exit|quit)\b', 
     ["Goodbye! Have a great day.", "See you later!"]),
     
    # Fallback rule: (.*) matches anything if no previous rules matched
    (r'(.*)', 
     ["I'm sorry, I didn't quite catch that.", "Could you rephrase that?", "Interesting, tell me more."]) 
]

def get_response(user_input):
    # Convert to lowercase for easier, case-insensitive matching
    user_input = user_input.lower()
    
    # Check each rule one by one
    for pattern, responses in RULES:
        # re.search looks for the pattern anywhere in the string
        if re.search(pattern, user_input):
            # Pick a random response from the matched category to feel more natural
            return random.choice(responses)

# --- Testing the code ---
if __name__ == "__main__":
    print("Chatbot initialized. Type 'quit' or 'bye' to exit.")
    print("-" * 50)
    
    while True:
        user_input = input("You: ")
        
        response = get_response(user_input)
        print(f"Bot: {response}")
        
        # Check if the user wants to exit
        if re.search(r'\b(bye|exit|quit)\b', user_input.lower()):
            break
```

**8\. Simple Reflex Agent (Vacuum Cleaner World)**

```py
# The Simple Reflex Agent logic
def simple_reflex_agent(location, status):
    if status == 'Dirty':
        return 'Suck'
    elif location == 'A':
        return 'Right'
    elif location == 'B':
        return 'Left'

def simulate_vacuum_world(starting_location, room_a_status, room_b_status):
    # Initialize the environment
    environment = {
        'A': room_a_status,
        'B': room_b_status
    }
    
    current_location = starting_location
    score = 0  # To analyze rational behavior (performance measure)
    
    print(f"Initial State: Room A is {environment['A']}, Room B is {environment['B']}")
    print(f"Vacuum starts in Room {current_location}\n")
    
    # Run the simulation for a fixed number of steps (e.g., 3 steps is enough for 2 rooms)
    for step in range(1, 4):
        print(f"--- Step {step} ---")
        
        # 1. Sense the environment (Percept)
        current_status = environment[current_location]
        print(f"Percept: [Location: {current_location}, Status: {current_status}]")
        
        # 2. Decide on an action based on rules
        action = simple_reflex_agent(current_location, current_status)
        print(f"Action chosen: {action}")
        
        # 3. Execute the action and update the environment/score
        if action == 'Suck':
            environment[current_location] = 'Clean'
            score += 10 # Reward for cleaning
            print(f"Room {current_location} is now Clean.")
            
        elif action == 'Right':
            current_location = 'B'
            score -= 1  # Penalty for moving
            
        elif action == 'Left':
            current_location = 'A'
            score -= 1  # Penalty for moving
            
        # Check if the whole environment is clean to stop early
        if environment['A'] == 'Clean' and environment['B'] == 'Clean':
            print("\nGoal Achieved: Both rooms are clean!")
            break
            
    print(f"\nFinal Performance Score: {score}")

# --- Testing the code ---
if __name__ == "__main__":
    # Test Case: Start in A. A is Dirty, B is Dirty.
    simulate_vacuum_world(starting_location='A', room_a_status='Dirty', room_b_status='Dirty')
```

&nbsp;
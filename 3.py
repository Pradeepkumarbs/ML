import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue

# ------------------------------------
# PART 1: Contour Plot Visualization
# ------------------------------------

# Generate sample n-dimensional data
X, Y = np.meshgrid(np.linspace(-3, 3, 100),
                   np.linspace(-3, 3, 100))

Z = np.sin(X**2 + Y**2)

plt.figure(figsize=(8, 6))

contour = plt.contourf(X, Y, Z, cmap='coolwarm')
plt.colorbar(contour)

plt.title("Contour Plot Visualization")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()

# ------------------------------------
# PART 2: A* Search Algorithm
# ------------------------------------

def a_star(graph, heuristics, start, goal):
    open_set = PriorityQueue()
    open_set.put((heuristics[start], start))

    g_cost = {node: float('inf') for node in graph}
    g_cost[start] = 0

    came_from = {}

    while not open_set.empty():
        _, current = open_set.get()

        if current == goal:
            path = [goal]

            while current in came_from:
                current = came_from[current]
                path.append(current)

            path.reverse()

            print("\nShortest Path:")
            print(" -> ".join(path))
            print("Total Cost:", g_cost[goal])
            return

        for neighbor, cost in graph[current]:
            new_cost = g_cost[current] + cost

            if new_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_cost
                priority = new_cost + heuristics[neighbor]

                open_set.put((priority, neighbor))
                came_from[neighbor] = current

# Example Graph
graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('D', 3)],
    'C': [('D', 1)],
    'D': []
}

heuristics = {
    'A': 4,
    'B': 2,
    'C': 2,
    'D': 0
}

# Run A* Algorithm
a_star(graph, heuristics, 'A', 'D')

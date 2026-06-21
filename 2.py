from queue import PriorityQueue
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# PART 1: 3D Surface Plot Visualization

np.random.seed(42)

# Generate sample n-dimensional data
n = 100
data = pd.DataFrame({
    'X': np.random.normal(0, 1, n),
    'Y': np.random.normal(0, 1, n),
    'Z': np.random.normal(0, 1, n)
})

print("Sample Dataset:")
print(data.head())

# Create 3D Surface Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
Z = np.sin(np.sqrt(X**2 + Y**2))

ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_title("3D Surface Plot")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()

# PART 2: Best First Search Algorithm

def best_first_search(graph, heuristics, start, goal):
    visited = set()
    pq = PriorityQueue()
    pq.put((heuristics[start], start))
    parent = {}

    while not pq.empty():
        _, current = pq.get()

        if current == goal:
            path = [goal]
            while current in parent:
                current = parent[current]
                path.append(current)

            path.reverse()
            print("\nPath Found:")
            print(" -> ".join(path))
            return

        if current not in visited:
            visited.add(current)

            for neighbor in graph[current]:
                if neighbor not in visited:
                    pq.put((heuristics[neighbor], neighbor))
                    parent[neighbor] = current

    print("Goal not reachable")

# Example Graph
graph = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['D'],
    'D': []
}

heuristics = {
    'A': 2,
    'B': 1,
    'C': 1,
    'D': 0
}

# Run Best First Search
best_first_search(graph, heuristics, 'A', 'D')

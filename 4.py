import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------
# PART 1: Heatmap Visualization
# -----------------------------------

data = np.random.rand(10, 10)

plt.figure(figsize=(8, 6))
sns.heatmap(data, cmap='viridis')
plt.title("Heatmap Visualization")
plt.show()

# -----------------------------------
# PART 2: Minimax Algorithm
# -----------------------------------

def minimax(depth, index, is_max, scores, max_depth):
    if depth == max_depth:
        return scores[index]

    left = minimax(depth + 1, index * 2,
                   not is_max, scores, max_depth)

    right = minimax(depth + 1, index * 2 + 1,
                    not is_max, scores, max_depth)

    if is_max:
        return max(left, right)
    else:
        return min(left, right)

# Example Game Tree
scores = [3, 5, 6, 9, 1, 2, 0, -1]
max_depth = 3

result = minimax(0, 0, True, scores, max_depth)

print("Minimax Optimal Value:", result)

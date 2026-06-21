import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Part 1: Visualization using Box Plot
def visualize_ndimensional_data():
    # Generate sample n-dimensional data (5 dimensions, 100 samples)
    data = np.random.randn(100, 5)

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data)
    plt.title("Box Plot for N-Dimensional Data")
    plt.xlabel("Dimensions")
    plt.ylabel("Values")
    plt.show()

# Part 2: Min-Max Algorithm
def minimax(depth, index, is_max, scores, max_depth):
    if depth == max_depth:
        return scores[index]

    left = minimax(depth + 1, index * 2, not is_max, scores, max_depth)
    right = minimax(depth + 1, index * 2 + 1, not is_max, scores, max_depth)

    return max(left, right) if is_max else min(left, right)

# Part 3: Alpha-Beta Pruning
def alphabeta(depth, index, is_max, scores, max_depth, alpha, beta):
    if depth == max_depth:
        return scores[index]

    if is_max:
        value = float('-inf')
        for i in range(2):
            value = max(
                value,
                alphabeta(depth + 1, index * 2 + i, False,
                          scores, max_depth, alpha, beta)
            )
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value
    else:
        value = float('inf')
        for i in range(2):
            value = min(
                value,
                alphabeta(depth + 1, index * 2 + i, True,
                          scores, max_depth, alpha, beta)
            )
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

# Main Program
if __name__ == "__main__":

    # Visualize n-dimensional data
    visualize_ndimensional_data()

    # Example game tree leaf values (depth = 3 → 8 leaves)
    scores = [3, 5, 6, 9, 1, 2, 0, -1]
    max_depth = 3

    # Min-Max Result
    minimax_result = minimax(0, 0, True, scores, max_depth)

    # Alpha-Beta Result
    alphabeta_result = alphabeta(
        0, 0, True, scores, max_depth,
        float('-inf'), float('inf')
    )

    print("Min-Max Optimal Value:", minimax_result)
    print("Alpha-Beta Optimal Value:", alphabeta_result)

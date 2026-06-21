# Hill Climbing Algorithm Example in Python

# Function to maximize
def objective_function(x):
    return -x**2 + 6*x


# Hill Climbing Algorithm
def hill_climbing(start, step_size=1, max_iterations=100):
    current = start
    current_value = objective_function(current)

    for _ in range(max_iterations):

        # Generate neighbors
        left_neighbor = current - step_size
        right_neighbor = current + step_size

        # Evaluate neighbors
        left_value = objective_function(left_neighbor)
        right_value = objective_function(right_neighbor)

        # Choose the better neighbor
        if left_value > current_value:
            current = left_neighbor
            current_value = left_value

        elif right_value > current_value:
            current = right_neighbor
            current_value = right_value

        else:
            # No better neighbor found
            break

    return current, current_value


# Starting point
start_point = 0

# Run Hill Climbing
best_x, best_value = hill_climbing(start_point)

# Output
print("Best x:", best_x)
print("Maximum value:", best_value)

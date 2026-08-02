import numpy as np
import matplotlib.pyplot as plt
import os

# The original function we are trying to minimize: f(x) = (x - 3)^2
def f_1d(x):
    return (x - 3)**2

def derivative_1d(x):
    return 2 * (x - 3)


# 1. The 2D Gradient Function
def gradient_2d(point):
    x, y = point
    return np.array([2 * x, 10 * y])


def gradient_descent_1d(start_x, lr, steps):
    x = start_x
    history = [x]

    for _ in range(steps):
        grad = derivative_1d(x)
        x = x - (lr * grad)
        history.append(x)

    return x, np.array(history)


# 2. The 2D Gradient Descent Loop
def gradient_descent_2d(start_point, lr, steps):
    point = np.array(start_point)
    history = [point]

    for _ in range(steps):
        grad = gradient_2d(point)
        point = point - (lr * grad)
        history.append(point)

    # Converts a list of [x, y] arrays into a single matrix of shape (steps, 2)
    return np.array(history)

def plot_gradient_descent(a,b,c):
    x_curve = np.linspace(-1, 7, 100)
    y_curve = (x_curve - 3) ** 2

    plt.figure(figsize=(10, 6))

    # Plot the actual U-shape curve in the background (thick black line)
    plt.plot(x_curve, y_curve, label="True Function: f(x) = (x-3)^2", color="black", linewidth=2, zorder=1)

    # 2. Get our gradient descent paths for the 3 rates
    # I swapped 1.0 for 1.05 so you can clearly see the "diverging" explosion
    rates = [a,b,c]
    colors = ['purple', 'orange', 'blue']

    for lr, color in zip(rates, colors):
        # Run gradient descent for this specific learning rate
        final_x, history_x = gradient_descent_1d(start_x=0.0, lr=lr, steps=10)

        # Calculate the height (loss) at each step to plot it on the curve
        loss_history = (history_x - 3) ** 2

        # Plot the path taken ON the U-curve
        plt.plot(history_x, loss_history, marker='o', linestyle='--', color=color, label=f"Path (LR = {lr})", zorder=2)

    plt.title("Gradient Descent Paths along the U-shaped Curve")
    plt.xlabel("x (Position)")
    plt.ylabel("f(x) (Height / Loss)")
    plt.legend()

    # We limit the y-axis so the exploding LR=1.05 doesn't zoom the graph out too far
    plt.ylim(-2, 20)
    plt.savefig('plots/gradient_descent.png', bbox_inches='tight')
    plt.show()


def plot_loss_function(a,b,c):
    rates = [a,b,c]
    colors = ['purple', 'orange', 'blue']
    start_x = 0.0

    # Kept to 10 steps so the diverging rate doesn't ruin the linear scale!
    steps = 50

    plt.figure(figsize=(8, 5))

    for lr, color in zip(rates, colors):
        # 1. Run gradient descent
        final_x, history_x = gradient_descent_1d(start_x, lr, steps)

        # 2. Calculate the loss at each step
        loss_history = (history_x - 3) ** 2

        # 3. Plot it
        plt.plot(loss_history, marker='o', color=color, label=f'LR = {lr}')

    plt.title("Loss vs. Steps (Convergence, Oscillation, and Divergence)")
    plt.xlabel("Step Number")
    plt.ylabel("Loss (Linear Scale)")
    plt.legend()

    plt.savefig('plots/Loss_function.png', bbox_inches='tight')
    # Look, no log scale!
    plt.show()

# 3. The Visualization Function
def plot_2d_gradient_descent(history_2d, lr, steps):
    plt.figure(figsize=(8, 6))

    # Generate the grid and topographical rings
    x_grid = np.linspace(-4, 4, 100)
    y_grid = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = X**2 + 5 * Y**2
    plt.contour(X, Y, Z, levels=20, cmap='viridis')

    # Extract X and Y paths
    path_x = history_2d[:, 0]
    path_y = history_2d[:, 1]

    # Plot the red path
    plt.plot(path_x, path_y, marker='o', color='red', label=f'GD Path (LR={lr})')

    plt.title(f"2D Gradient Descent (LR={lr}, Steps={steps})")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.legend()
    plt.savefig('plots/2D_gradient_descent.png', bbox_inches='tight')
    plt.show()
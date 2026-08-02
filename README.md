# Machine Learning Basics from Scratch

This project contains simple, easy-to-understand Python implementations of three fundamental Machine Learning algorithms. The code is written in plain English to help beginners understand how these models work behind the scenes, complete with helpful visual graphs.

## What's Included?

### 1. Gradient Descent
Gradient descent is the optimization method that helps machine learning models "learn." 
* **1D and 2D Gradient Descent**: The code includes functions that take step-by-step jumps down a mathematical curve to find the lowest point, which represents the minimum error or loss.
* **Visualizations**: It includes graphing tools to draw the true U-shaped curve and plot the exact path the algorithm takes to reach the bottom. It also graphs what happens when your "learning rate" (step size) is too big and the model explodes or diverges on a linear scale.

### 2. K-Nearest Neighbors (KNN)
KNN is a simple algorithm used to classify data. To figure out what category a new point belongs to, it simply looks at its closest neighbors.
* **Distance Calculators**: The code provides functions to measure the distance between points using Cosine Similarity and Euclidean Distance.
* **KNN Predictor**: It includes a function that finds the 'k' closest training data points to a test point and takes a vote to decide its classification.
* **Visualizations**: It features a plotting function that draws a colorful decision boundary map showing how different areas are classified, and highlights your specific external query point as a large yellow star.

### 3. Principal Component Analysis (PCA)
PCA is a technique used to simplify complex data by reducing its dimensions while keeping the most important patterns.
* **PCA via SVD**: The code includes a function that centers the data and uses Singular Value Decomposition (SVD) to find the principal components of the dataset.
* **Visualizations**: It features a plotting tool that projects the data down onto the principal direction line and graphs it against the original scattered data so you can see exactly how the dimension reduction works.


## Automatically Saving Plots
Whenever you run any of the visualization functions (like `plot_gradient_descent`, `plot_knn_prediction`, or `plot_pca_reconstruction`), the code will automatically create a `plots` folder in your current directory and save a high-quality `.png` image of the graph there. This makes it easy to keep a record of your experiments without manually saving each figure!

## Requirements
* NumPy
* Matplotlib
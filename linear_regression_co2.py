# linear_regression_co2.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter


df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "rinichristy/2022-fuel-consumption-ratings",
    "MY2022 Fuel Consumption Ratings.csv",
)
df = df[["Engine Size(L)", "Cylinders", "CO2 Emissions(g/km)"]]

X = df[["Engine Size(L)", "Cylinders"]].values
y = df["CO2 Emissions(g/km)"].values

m = len(y)
X_b = np.c_[np.ones((m, 1)), X]
y = y.reshape(-1, 1)

def compute_cost(X, y, theta):
    """Compute mean squared error cost"""
    m = len(y)
    predictions = X.dot(theta)
    error = predictions - y
    cost = (1/(2*m)) * np.sum(error ** 2)
    return cost

def gradient_descent(X, y, theta, alpha, epochs):
    """Perform gradient descent"""
    m = len(y)
    cost_history = []

    for i in range(epochs):
        gradients = (1/m) * X.T.dot(X.dot(theta) - y)
        theta -= alpha * gradients
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

theta = np.zeros((X_b.shape[1], 1)) 
alpha = 0.00001
epochs = 20000

print("Starting values of parameters (theta):", theta.ravel())
print("Starting learning rate (alpha):", alpha)
print("Starting epochs:", epochs)
print("Starting cost:", compute_cost(X_b, y, theta))

theta_final, cost_history = gradient_descent(X_b, y, theta, alpha, epochs)

print("\nFinal parameters (theta):", theta_final.ravel())
print("Final cost:", cost_history[-1])

plt.plot(range(epochs), cost_history, "b-")
plt.xlabel("Epochs")
plt.ylabel("Cost (MSE)")
plt.title("Cost Function Convergence")
plt.show()

engine_size_range = np.linspace(df["Engine Size(L)"].min(), df["Engine Size(L)"].max(), 100)
cyl_fixed = 4 
X_plot = np.c_[np.ones(100), engine_size_range, np.full(100, cyl_fixed)]
y_pred = X_plot.dot(theta_final)

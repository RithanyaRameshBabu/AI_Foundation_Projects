

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create synthetic binary classification data
np.random.seed(0)

X = np.linspace(0, 10, 100)
y = (X > 5).astype(int)  # Class 0 if x<=5, Class 1 if x>5

# Step 2: Initialize parameters
w = 0.0
b = 0.0
learning_rate = 0.1
epochs = 1000
n = len(X)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Binary cross-entropy loss
def loss_function(y, y_pred):
    epsilon = 1e-9  # to avoid log(0)
    return -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))

losses = []

# Step 3: Gradient Descent
for _ in range(epochs):
    z = w * X + b
    y_pred = sigmoid(z)

    loss = loss_function(y, y_pred)
    losses.append(loss)

    dw = np.mean((y_pred - y) * X)
    db = np.mean(y_pred - y)

    w -= learning_rate * dw
    b -= learning_rate * db

# Step 4: Final output
print("Final weight (w):", round(w, 2))
print("Final bias (b):", round(b, 2))

# Step 5: Plot decision boundary
plt.scatter(X, y, label="Actual data")
plt.plot(X, sigmoid(w * X + b), color="red", label="Predicted probability")
plt.axhline(0.5, color="gray", linestyle="--")
plt.xlabel("X")
plt.ylabel("Probability / Class")
plt.legend()
plt.show()

# Step 6: Plot loss curve
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.show()
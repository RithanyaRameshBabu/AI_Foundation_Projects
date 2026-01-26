
# ------------------------------
# Linear Regression
# ------------------------------

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Step 1: Create synthetic data
# ------------------------------

# Setting seed for reproducibility
np.random.seed(42)

# Generate 50 data points between 1 and 10
X = np.linspace(1, 10, 50)

# True relationship y = 3*X + 5 plus random noise
y = 3 * X + 5 + np.random.randn(50) * 2

# First-person reflection:
# I studied how Linear Regression predicts numeric outcomes.
# I learned that gradient descent updates weights iteratively to minimize error.

# ------------------------------
# Step 2: Initialize parameters
# ------------------------------

w = 0.0  # slope (weight)
b = 0.0  # intercept (bias)
learning_rate = 0.01
epochs = 1000
n = len(X)  # number of data points

# ------------------------------
# Step 3: Gradient Descent Loop
# ------------------------------

losses = []

for epoch in range(epochs):
    # Predicted values
    y_pred = w * X + b

    # Mean Squared Error Loss
    loss = np.mean((y - y_pred) ** 2)
    losses.append(loss)

    # Gradients
    dw = (-2 / n) * np.sum(X * (y - y_pred))
    db = (-2 / n) * np.sum(y - y_pred)

    # Update weights
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Optional: Print loss every 100 epochs
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

# ------------------------------
# Step 4: Display final parameters
# ------------------------------

print("\nFinal Learned Parameters:")
print("Weight (w):", round(w, 2))
print("Bias (b):", round(b, 2))

# ------------------------------
# Step 5: Plot regression line
# ------------------------------

plt.figure(figsize=(8,5))
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, w * X + b, color='red', label='Predicted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()

# ------------------------------
# Step 6: Plot Loss vs Epochs
# ------------------------------

plt.figure(figsize=(8,5))
plt.plot(losses, color='green')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Loss vs Epochs')
plt.show()


# 4. Applications include house price prediction, sales forecasting, and student score prediction.

import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Step 1: Generate synthetic dataset
np.random.seed(0)
mean0, mean1 = [-1, -1], [1, 1]
cov = np.eye(2)

X0 = np.random.multivariate_normal(mean0, cov, 10)
X1 = np.random.multivariate_normal(mean1, cov, 10)

X = np.vstack((X0, X1))
y = np.array([0]*10 + [1]*10).reshape(-1, 1)

# Shuffle and split into 50% train and 50% test
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

split = len(X) // 2
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 2: Initializing weights and biases randomly
W1 = np.random.randn(2, 1)
b1 = np.random.randn(1)
W2 = np.random.randn(1, 1)
b2 = np.random.randn(1)

lr = 0.1
epochs = 1000
losses = []

# Step 3: Training using gradient descent
for epoch in range(epochs):
    # Forward Pass
    Z1 = X_train @ W1 + b1              # Hidden layer linear
    A1 = sigmoid(Z1)                    # Hidden layer activation
    Z2 = A1 @ W2 + b2                   # Output layer (linear)
    y_pred = Z2                         # Final prediction

    # Compute Loss (MSE)
    loss = np.mean((y_pred - y_train) ** 2)
    losses.append(loss)

    # Backpropagation
    dZ2 = 2 * (y_pred - y_train) / y_train.shape[0]   # dL/dZ2
    dW2 = A1.T @ dZ2                                  # dL/dW2
    db2 = np.sum(dZ2)                                 # dL/db2

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * sigmoid_derivative(Z1)                # dL/dZ1
    dW1 = X_train.T @ dZ1                             # dL/dW1
    db1 = np.sum(dZ1)                                 # dL/db1

    # Parameter Update
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d}: Train MSE = {loss:.4f}")

Z1_test = X_test @ W1 + b1
A1_test = sigmoid(Z1_test)
Z2_test = A1_test @ W2 + b2
y_test_pred = Z2_test

mse_test = np.mean((y_test_pred - y_test) ** 2)
print(f"\nTest MSE: {mse_test:.4f}")

plt.plot(losses)
plt.title("Training MSE over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

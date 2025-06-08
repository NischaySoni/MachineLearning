import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Generate data
np.random.seed(42)
x = np.random.uniform(0, 1, 100)
epsilon = np.random.normal(0, 0.01, 100)
y = np.sin(2 * np.pi * x) + np.cos(2 * np.pi * x) + epsilon

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Reshape
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

# Fitting decision stump on the negative gradients.
def train_stump_regression(x, y):
    best_threshold = None
    best_left_value = None
    best_right_value = None
    min_loss = float('inf')
    thresholds = np.linspace(0, 1, 21)[1:-1]  # 20 cuts

    for threshold in thresholds:
        left_mask = x[:, 0] < threshold
        right_mask = ~left_mask
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            continue

        left_value = np.mean(y[left_mask])
        right_value = np.mean(y[right_mask])
        preds = np.where(left_mask, left_value, right_value)
        loss = np.mean((y - preds) ** 2)

        if loss < min_loss:
            min_loss = loss
            best_threshold = threshold
            best_left_value = left_value
            best_right_value = right_value

    return {
        'threshold': best_threshold,
        'left_value': best_left_value,
        'right_value': best_right_value
    }

def predict_gb(x, models, learning_rate=0.01):
    preds = np.zeros(x.shape[0])
    for stump in models:
        stump_preds = np.where(x[:, 0] < stump['threshold'], stump['left_value'], stump['right_value'])
        preds += learning_rate * stump_preds
    return preds

# Computation of negative gradients for each loss function.
def gradient_boosting_regression(x, y, x_test, y_test, loss_type='squared', n_rounds=100, learning_rate=0.01):
    models = []
    predictions = np.zeros_like(y)
    test_predictions = np.zeros_like(y_test)

    train_losses = []
    test_losses = []

    intermediate_preds_train = {}
    intermediate_preds_test = {}

    for i in range(n_rounds):
        # Compute residuals
        if loss_type == 'squared':
            residuals = y - predictions
        elif loss_type == 'absolute':
            residuals = np.sign(y - predictions)

        stump = train_stump_regression(x, residuals)
        stump_preds_train = np.where(x[:, 0] < stump['threshold'], stump['left_value'], stump['right_value'])
        stump_preds_test = np.where(x_test[:, 0] < stump['threshold'], stump['left_value'], stump['right_value'])

# Final Prediction as Sum of Weak Learners
        predictions += learning_rate * stump_preds_train
        test_predictions += learning_rate * stump_preds_test

        if loss_type == 'squared':
            train_loss = np.mean((y - predictions) ** 2)
            test_loss = np.mean((y_test - test_predictions) ** 2)
        else:
            train_loss = np.mean(np.abs(y - predictions))
            test_loss = np.mean(np.abs(y_test - test_predictions))

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        models.append(stump)

        if i % 10 == 0 or i == n_rounds - 1:
            intermediate_preds_train[i] = predictions.copy()
            intermediate_preds_test[i] = test_predictions.copy()
            print(f"[{loss_type}] Iteration {i+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    return models, train_losses, test_losses, intermediate_preds_train, intermediate_preds_test

# Training for both loss types
models_sq, train_losses_sq, test_losses_sq, preds_train_steps_sq, preds_test_steps_sq = gradient_boosting_regression(
    x_train, y_train, x_test, y_test, loss_type='squared'
)
models_abs, train_losses_abs, test_losses_abs, preds_train_steps_abs, preds_test_steps_abs = gradient_boosting_regression(
    x_train, y_train, x_test, y_test, loss_type='absolute'
)

# Plot evolution of predictions
def plot_prediction_evolution(x_train, y_train, x_test, y_test, steps_dict_train, steps_dict_test, loss_name):
    x_lin = np.linspace(0, 1, 100)
    y_true = np.sin(2 * np.pi * x_lin) + np.cos(2 * np.pi * x_lin)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(f"Prediction Evolution - {loss_name.capitalize()} Loss", fontsize=16)

    steps_to_plot = sorted(steps_dict_test.keys())[:6]

    for idx, step in enumerate(steps_to_plot):
        ax = axes[idx // 3, idx % 3]
        ax.scatter(x_lin, y_true, label='True Function', color='green', s=15)
        ax.scatter(x_test, steps_dict_test[step], label='Test Prediction', color='red', s=15)
        ax.set_title(f"Iteration {step+1}")
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()

# Plotting both
plot_prediction_evolution(x_train, y_train, x_test, y_test, preds_train_steps_sq, preds_test_steps_sq, 'squared')
plot_prediction_evolution(x_train, y_train, x_test, y_test, preds_train_steps_abs, preds_test_steps_abs, 'absolute')

# Plot training loss over iterations
plt.figure(figsize=(10, 5))
plt.plot(train_losses_sq, label='Train Loss (Squared)')
plt.plot(train_losses_abs, label='Train Loss (Absolute)')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training loss over Iterations")
plt.legend()
plt.grid()
plt.show()

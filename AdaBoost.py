import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# implementing decision stump
class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.beta = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)

        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column >= self.threshold] = -1

        return predictions

# implementing adaboost
class AdaBoost:
    def __init__(self, n_estimators=200):
        self.n_estimators = n_estimators
        self.stumps = []
        self.train_errors = []
        self.val_errors = []
        self.test_errors = []

    def fit(self, X, y, X_val=None, y_val=None, X_test=None, y_test=None):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))
        self.betas = []

# Training loop that updates sample weights based on misclassification
        for _ in range(self.n_estimators):
            stump = self._train_stump(X, y, w)
            # Calculation of error using weighted 0-1 loss and classifier weights β
            predictions = stump.predict(X)
            err = np.sum(w * (predictions != y)) / np.sum(w)
            beta = 0.5 * np.log((1 - err) / (max(err, 1e-10)))
            self.betas.append(beta)
            w = w * np.exp(-beta * y * predictions)
            w /= np.sum(w)

            stump.beta = beta
            self.stumps.append(stump)

            self.train_errors.append(self._calculate_error(X, y))

            if X_val is not None and y_val is not None:
                self.val_errors.append(self._calculate_error(X_val, y_val))

            if X_test is not None and y_test is not None:
                self.test_errors.append(self._calculate_error(X_test, y_test))

    def _train_stump(self, X, y, w):
        stump = DecisionStump()
        min_error = float('inf')
        n_features = X.shape[1]

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            min_val, max_val = np.min(feature_values), np.max(feature_values)

            # For each decision stump, evaluate 3 (here we are taking 8) cuts uniformly spaced per dimension
            thresholds = np.linspace(min_val, max_val, 10)[1:-1]  # More thresholds for better splits

            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(X.shape[0])
                    if polarity == 1:
                        predictions[feature_values < threshold] = -1
                    else:
                        predictions[feature_values >= threshold] = -1

                    error = np.sum(w * (predictions != y)) / np.sum(w)

                    if error < min_error:
                        min_error = error
                        stump.polarity = polarity
                        stump.threshold = threshold
                        stump.feature_idx = feature_idx
        return stump

# Final prediction rule based on boosted classifier
    def predict(self, X):
        n_samples = X.shape[0]
        result = np.zeros(n_samples)

        for stump in self.stumps:
            predictions = stump.predict(X)
            result += stump.beta * predictions

        return np.sign(result)

    def _calculate_error(self, X, y):
        predictions = self.predict(X)
        return 1 - accuracy_score(y, predictions)

def plot_errors(adaboost):
    plt.figure(figsize=(10, 6))

    # Plot for Train/Val/Test errors
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(adaboost.train_errors) + 1), adaboost.train_errors, label='Training error')

    if adaboost.val_errors:
        plt.plot(range(1, len(adaboost.val_errors) + 1), adaboost.val_errors, label='Validation error')

    if adaboost.test_errors:
        plt.plot(range(1, len(adaboost.test_errors) + 1), adaboost.test_errors, label='Test error')

    plt.xlabel('Number of boosting rounds')
    plt.ylabel('Error rate')
    plt.title('Train/Val/Test Error as a function of Boosting Rounds')
    plt.legend()
    plt.grid(True)

    # Plot for Training Error as a function of Boosting Rounds
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(adaboost.train_errors) + 1), adaboost.train_errors, label='Training error')

    plt.xlabel('Number of boosting rounds')
    plt.ylabel('Error rate')
    plt.title('Training Error as a function of Boosting Rounds')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Use only 1000 samples per class for training
def sample_balanced(X, y, samples_per_class=1000):
    class_neg_indices = np.where(y == -1)[0]
    class_pos_indices = np.where(y == 1)[0]
    sampled_neg = np.random.choice(class_neg_indices, samples_per_class, replace=False)
    sampled_pos = np.random.choice(class_pos_indices, samples_per_class, replace=False)
    sampled_indices = np.concatenate([sampled_neg, sampled_pos])
    return X[sampled_indices], y[sampled_indices]

def main():
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=True)
    X = mnist.data.to_numpy()
    y = mnist.target.astype(int).to_numpy()

# Evaluation on MNIST using classes 0 and 1 only
    # Filter for digits 0 and 1
    mask = (y == 0) | (y == 1)
    X, y = X[mask], y[mask]
    y = np.where(y == 0, -1, 1)

    # Train/test/validation split
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42)

    # Sample 1000 of each class from each set
    X_train_sampled, y_train_sampled = sample_balanced(X_train, y_train)
    X_val_sampled, y_val_sampled = sample_balanced(X_val, y_val)
    X_test_sampled, y_test_sampled = sample_balanced(X_test, y_test)

# Apply PCA to reduce to 5 dimensions
    print("Applying PCA...")
    pca = PCA(n_components=5)
    X_train_pca = pca.fit_transform(X_train_sampled)
    X_val_pca = pca.transform(X_val_sampled)
    X_test_pca = pca.transform(X_test_sampled)

    # Train AdaBoost
    print("Training AdaBoost...")
    adaboost = AdaBoost(n_estimators=200)
    adaboost.fit(X_train_pca, y_train_sampled, X_val_pca, y_val_sampled, X_test_pca, y_test_sampled)

# Test on full test set of both classes
    test_pred = adaboost.predict(X_test_pca)
    accuracy = accuracy_score(y_test_sampled, test_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Plot errors
    plot_errors(adaboost)

if __name__ == "__main__":
    main()

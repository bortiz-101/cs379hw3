import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics import f1_score

# Load dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# Preprocessing function for both train and dev data
def preprocess_data(df):
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    return df


# Apply preprocessing to train and dev data
train_data = preprocess_data(train)
test_data = preprocess_data(test)

# Set features and target variable
X_train = train_data[['Pclass', 'Sex', 'Age', 'Fare']].values
y_train = train_data['Survived'].values
X_test = test_data[['Pclass', 'Sex', 'Age', 'Fare']].values
y_test = test_data['Survived'].values

# Standardize features
scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.transform(X_test)


class KNNClassifier:
    def __init__(self, k=10):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X_dev):
        predictions = [self._predict(x) for x in X_dev]
        return np.array(predictions)

    def _predict(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get k nearest samples
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return the most common label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


# Document the results
with open('Final_KNN_results.txt', 'w') as f:
    knn = KNNClassifier(9)
    knn.fit(X_train_standardized, y_train)

    # Predict on test
    y_test_pred = knn.predict(X_test_standardized)

    # Evaluate performance using F1 score
    f1_test = f1_score(y_test, y_test_pred)

    # Write F1 score file
    f.write(f"F1 Score on Test Set: {f1_test:.2f}")

    # Print progress to console
    print(f"F1 Score on Development Set: {f1_test:.2f}")

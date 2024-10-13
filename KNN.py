import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics import f1_score

# Load dataset
train = pd.read_csv("train.csv")
dev = pd.read_csv("dev.csv")


# Preprocessing function for both train and dev data
def preprocess_data(df):
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    return df


# Apply preprocessing to train and dev data
train_data = preprocess_data(train)
dev_data = preprocess_data(dev)

# Set features and target variable
X_train = train_data[['Pclass', 'Sex', 'Age', 'Fare']].values
y_train = train_data['Survived'].values
X_dev = dev_data[['Pclass', 'Sex', 'Age', 'Fare']].values
y_dev = dev_data['Survived'].values

# Standardize features
scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train)
X_dev_standardized = scaler.transform(X_dev)


class KNNClassifier:
    def __init__(self, k=3):
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


# Document the results for each k value from 1 to 10
with open('KNN_tuning_results.txt', 'w') as f:
    for k in range(1, 11):
        knn = KNNClassifier(k=k)
        knn.fit(X_train_standardized, y_train)

        # Predict on dev
        y_dev_pred = knn.predict(X_dev_standardized)

        # Evaluate performance using F1 score
        f1_dev = f1_score(y_dev, y_dev_pred)

        # Write F1 score file
        f.write(f"K={k}: F1 Score on Development Set: {f1_dev:.2f}\n")


        # Print progress to console
        print(f"K={k}: F1 Score on Development Set: {f1_dev:.2f}")



import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

# Load dataset
train = pd.read_csv("train.csv")
dev = pd.read_csv("dev.csv")


# Preprocessing function
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


# Function to evaluate DummyClassifier with different strategies
def evaluate_dummy(strategy):
    dummy_clf = DummyClassifier(strategy=strategy, random_state=42)
    dummy_clf.fit(X_train, y_train)
    y_dev_pred = dummy_clf.predict(X_dev)

    f1_dev = f1_score(y_dev, y_dev_pred)
    print(f"F1 Score on Development Set with strategy '{strategy}': {f1_dev:.2f}")

    # Document the results
    with open('DummyClassifier_results.txt', 'a') as f:
        f.write(f"Strategy: {strategy}, F1 Score: {f1_dev:.2f}\n")


# Evaluate with 'stratified' strategy
evaluate_dummy(strategy='stratified')

# Evaluate with 'most_frequent' strategy
evaluate_dummy(strategy='most_frequent')

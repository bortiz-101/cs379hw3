import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# Load dataset
train = pd.read_csv("train.csv")
dev = pd.read_csv("dev.csv")


# Preprocessing function
def preprocess_data(df):
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    return df


# Apply preprocessing to train and test data
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

# Initialize and train the SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_standardized, y_train)

# Save the trained model to a file
model_filename = 'svm_model_weights.joblib'
joblib.dump(svm_model, model_filename)

# Make predictions on the train and dev data
predictions_train = svm_model.predict(X_train_standardized)
predictions_dev = svm_model.predict(X_dev_standardized)

# Save predictions to CSV files
train_output = pd.DataFrame({'PassengerId': train_data.index, 'Survived': predictions_train})
train_output.to_csv('predictions_train.csv', index=False)

dev_output = pd.DataFrame({'PassengerId': dev_data.index, 'Survived': predictions_dev})
dev_output.to_csv('predictions_dev.csv', index=False)

# Calculate and display survival rates
print(f"\nOverall Survival Rate on Training Data: {y_train.mean():.2f}")

# Evaluate model performance
f1_train = f1_score(y_train, predictions_train)
f1_dev = f1_score(y_dev, predictions_dev)

print(f"\nF1 Score on Training Data: {f1_train:.2f}")
print(f"F1 Score on Development Data: {f1_dev:.2f}")

import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# Load dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# Preprocessing function
def preprocess_data(df):
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    return df


# Apply preprocessing to train and test data
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

# Best Parameters: {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}
# Initialize and train the SVM model
svm_model = SVC(C=10, gamma='auto', kernel='rbf')
svm_model.fit(X_train_standardized, y_train)

# Save the trained model to a file
model_filename = 'Final_svm_model_weights.joblib'
joblib.dump(svm_model, model_filename)

# Make predictions on the train and dev data
predictions_train = svm_model.predict(X_train_standardized)
predictions_test = svm_model.predict(X_test_standardized)

# Calculate and display survival rates
print(f"\nOverall Survival Rate on Training Data: {y_train.mean():.2f}")
print(f"\nOverall Survival Rate on Test Data: {y_test.mean():.2f}")

# Evaluate model performance
f1_train = f1_score(y_train, predictions_train)
f1_test = f1_score(y_test, predictions_test)
with open('Final_SVM_results.txt', 'w') as f:
    f.write(f"\nOverall Survival Rate on Training Data: {y_train.mean():.2f}\n")
    f.write(f"\nOverall Survival Rate on Test Data: {y_test.mean():.2f}\n")
    f.write(f"\nF1 Score on Training Data: {f1_train:.2f}\n")
    f.write(f"F1 Score on Test Data: {f1_test:.2f}\n")

print(f"\nF1 Score on Training Data: {f1_train:.2f}")
print(f"F1 Score on Test Data: {f1_test:.2f}")

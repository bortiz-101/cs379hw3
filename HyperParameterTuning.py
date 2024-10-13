import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
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

# Apply preprocessing
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

# Define the SVM model
svm_model = SVC(random_state=42)

# Define the hyperparameters to tune
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Define F1 score as the scoring metric
f1_scorer = make_scorer(f1_score, average='binary')

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, scoring=f1_scorer, cv=5, verbose=2)

# Perform the grid search
grid_search.fit(X_train_standardized, y_train)

# Save the best model to a file
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'best_svm_model_weights.joblib')

# Evaluate on the development set using the best model
dev_predictions = best_model.predict(X_dev_standardized)
f1_dev = f1_score(y_dev, dev_predictions)

# Document the results
with open('hyperparameter_tuning_results.txt', 'w') as f:
    f.write(f"Best Parameters: {grid_search.best_params_}\n")
    f.write(f"Best F1 Score on Training Data (CV): {grid_search.best_score_:.4f}\n")
    f.write(f"F1 Score on Development Data: {f1_dev:.4f}\n\n")
    f.write("Grid Search Results:\n")
    for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
        f.write(f"F1 Score: {mean_score:.4f} with Parameters: {params}\n")

print("Hyperparameter tuning complete. Results saved to 'hyperparameter_tuning_results.txt'.")



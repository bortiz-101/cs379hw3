import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
data = "MBA.csv"
mba_data = pd.read_csv(data)

# Rename columns
mba_data.columns = ['application_id', 'gender', 'international', 'gpa', 'major', 'race', 'gmat',
                    'work_exp', 'work_industry', 'admission']

# Check unique application IDs
print('application_ids:', np.unique(mba_data['application_id']))

# Define features and target
X = mba_data.iloc[:, 1:].values  # Features (exclude application_id)
y = mba_data.iloc[:, -1].values

# Step 1: First split to create train (70%) and temp (30%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)

# Step 2: Split temp into dev (15%) and test (15%) from the original data
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Save the processed datasets to CSV
X_train.to_csv('X_train.csv', index=False)
X_dev.to_csv('X_dev.csv', index=False)
X_test.to_csv('X_test.csv', index=False)

# Save the target labels
y_train.to_csv('y_train.csv', index=False)
y_dev.to_csv('y_dev.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
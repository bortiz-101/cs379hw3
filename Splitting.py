import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = "MBA.csv"
mba_data = pd.read_csv(data)

# Rename columns
mba_data.columns = ['application_id', 'gender', 'international', 'gpa', 'major', 'race', 'gmat',
                    'work_exp', 'work_industry', 'admission']

# Separate rows with NaN 'admission' (for dev/test sets)
mba_data_nan = mba_data[mba_data['admission'].isna()]

# Keep rows with valid 'admission' values (for training)
mba_data_valid = mba_data.dropna(subset=['admission'])

# Split valid data (for training): Features (X) and Target (y)
X_train = mba_data_valid.iloc[:, 1:-1].values  # Features (exclude 'application_id' and 'admission')
y_train = mba_data_valid['admission'].values  # Target

# Use the NaN rows for dev and test sets
X_dev_test = mba_data_nan.iloc[:, 1:-1].values  # Features for NaN rows
y_dev_test = mba_data_nan['admission'].values  # Target for NaN rows (NaNs)

# Split the NaN rows into dev (50%) and test (50%)
X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test, test_size=0.5, random_state=42)

# Convert NumPy arrays to DataFrames for saving to CSV
X_train_df = pd.DataFrame(X_train, columns=mba_data.columns[1:-1])
X_dev_df = pd.DataFrame(X_dev, columns=mba_data.columns[1:-1])
X_test_df = pd.DataFrame(X_test, columns=mba_data.columns[1:-1])

y_train_df = pd.DataFrame(y_train, columns=['admission'])
y_dev_df = pd.DataFrame(y_dev, columns=['admission'])
y_test_df = pd.DataFrame(y_test, columns=['admission'])

# Save the processed datasets to CSV
X_train_df.to_csv('X_train.csv', index=False)
X_dev_df.to_csv('X_dev.csv', index=False)
X_test_df.to_csv('X_test.csv', index=False)

# Save the target labels to CSV
y_train_df.to_csv('y_train.csv', index=False)
y_dev_df.to_csv('y_dev.csv', index=False)
y_test_df.to_csv('y_test.csv', index=False)

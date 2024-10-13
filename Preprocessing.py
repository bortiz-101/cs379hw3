import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
data = "MBA.csv"
mba_data = pd.read_csv(data)

# Separate rows with and without admission status
known_admission_data = mba_data.dropna(subset=['admission'])
unknown_admission_data = mba_data[mba_data['admission'].isna()]

# Fill missing values in 'race' column using .loc to avoid SettingWithCopyWarning
known_admission_data.loc[:, 'race'] = known_admission_data['race'].fillna('Unknown')
unknown_admission_data.loc[:, 'race'] = unknown_admission_data['race'].fillna('Unknown')

# Map 'admission' column to numerical values
admission_mapping = {'Admit': 1, 'Waitlist': 0.5, 'Deny': 0}
known_admission_data.loc[:, 'admission'] = known_admission_data['admission'].map(admission_mapping)

# Define features and target for known admission data
X_known = known_admission_data.drop(columns=['application_id', 'admission'])
y_known = known_admission_data['admission']

# Define features for unknown admission data (this will be used for testing)
X_unknown = unknown_admission_data.drop(columns=['application_id', 'admission'])

# Identify categorical and numerical features
categorical_features = ['gender', 'international', 'major', 'race', 'work_industry']
numerical_features = ['gpa', 'gmat', 'work_exp']

# Create transformers for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Split known data into training (70%), development (15%), and test (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X_known, y_known, test_size=0.3, random_state=42, stratify=y_known)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Create a pipeline for preprocessing
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit the pipeline on the training data
pipeline.fit(X_train)

# Transform the training, development, test, and unknown admission sets
X_train_processed = pipeline.transform(X_train)
X_dev_processed = pipeline.transform(X_dev)
X_test_processed = pipeline.transform(X_test)
X_unknown_processed = pipeline.transform(X_unknown)

# Convert processed data back into DataFrame for in-memory use
X_train_df = pd.DataFrame(X_train_processed.toarray() if hasattr(X_train_processed, 'toarray') else X_train_processed)
X_dev_df = pd.DataFrame(X_dev_processed.toarray() if hasattr(X_dev_processed, 'toarray') else X_dev_processed)
X_test_df = pd.DataFrame(X_test_processed.toarray() if hasattr(X_test_processed, 'toarray') else X_test_processed)
X_unknown_df = pd.DataFrame(X_unknown_processed.toarray() if hasattr(X_unknown_processed, 'toarray') else X_unknown_processed)

# Save the processed datasets to CSV
X_train_df.to_csv('X_train_processed.csv', index=False)
X_dev_df.to_csv('X_dev_processed.csv', index=False)
X_test_df.to_csv('X_test_processed.csv', index=False)

# Save the target labels
y_train.to_csv('y_train.csv', index=False)
y_dev.to_csv('y_dev.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Example of accessing in-memory DataFrames
print("Shapes of processed datasets in memory:")
print("Training set shape:", X_train_df.shape)
print("Development set shape:", X_dev_df.shape)
print("Test set shape:", X_test_df.shape)
print("Unknown admission set shape:", X_unknown_df.shape)
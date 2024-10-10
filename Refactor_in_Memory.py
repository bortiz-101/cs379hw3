import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
url = "path_to_your_dataset/MBA.csv"  # Replace with the path to your local dataset
mba_data = pd.read_csv(url)

# Drop rows with missing target (admission) values
mba_data.dropna(subset=['admission'], inplace=True)

# Fill missing values in categorical columns without chained assignment
mba_data['race'] = mba_data['race'].fillna('Unknown')

# Define features and target
X = mba_data.drop(columns=['application_id', 'admission'])
y = mba_data['admission']

# Identify categorical and numerical features
categorical_features = ['gender', 'international', 'major', 'race', 'work_industry']
numerical_features = ['gpa', 'gmat', 'work_exp']

# Create transformers for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Split data into training (70%), development (15%), and test (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Create a pipeline for preprocessing
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit the pipeline on the training data
pipeline.fit(X_train)

# Transform the training, development, and test sets
X_train_processed = pipeline.transform(X_train)
X_dev_processed = pipeline.transform(X_dev)
X_test_processed = pipeline.transform(X_test)

# Convert processed data back into DataFrame for in-memory use
X_train_df = pd.DataFrame(X_train_processed.toarray() if hasattr(X_train_processed, 'toarray') else X_train_processed)
X_dev_df = pd.DataFrame(X_dev_processed.toarray() if hasattr(X_dev_processed, 'toarray') else X_dev_processed)
X_test_df = pd.DataFrame(X_test_processed.toarray() if hasattr(X_test_processed, 'toarray') else X_test_processed)

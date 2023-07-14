import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer, precision_score, recall_score, f1_score, classification_report
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Read the data
fault_data = pd.read_csv("/home/peverell/Documents/ds/selfmade_efe_mert_cem/data/fault_data.csv")
status_data = pd.read_csv("/home/peverell/Documents/ds/selfmade_efe_mert_cem/data/status_data.csv")
scada_data = pd.read_csv("/home/peverell/Documents/ds/selfmade_efe_mert_cem/data/scada_data.csv")

# Convert the DateTime columns to a common datetime format
scada_data['DateTime'] = pd.to_datetime(scada_data['DateTime'])
status_data['Time'] = pd.to_datetime(status_data['Time'], dayfirst=True)
status_data.rename(columns={'Time': 'DateTime'}, inplace=True)
fault_data['DateTime'] = pd.to_datetime(fault_data['DateTime'])

df_combine = scada_data.merge(fault_data, on='Time', how='outer')

df_combine['Fault'] = df_combine['Fault'].replace(np.nan, 'NF')

# Create a DataFrame with a random sample of 300 'NF' instances
df_nf = df_combine[df_combine.Fault == 'NF'].sample(300, random_state=42)

# Create a DataFrame with a random sample of 300 fault instances
df_fault = df_combine[df_combine.Fault != 'NF'].sample(300, random_state=42)

# Combine the two DataFrames
df_combine = pd.concat([df_nf, df_fault], axis=0).reset_index(drop=True)

# Reset the index of the concatenated DataFrame
df_combine.reset_index(drop=True, inplace=True)

pd.set_option('display.float_format', lambda x: '%.3f' % x)
df_summary = df_combine.groupby('Fault').mean().T

# Drop irrelevant features
train_df = df_combine.drop(columns=['DateTime_x', 'Time', 'Error', 'WEC: ava. windspeed', 
                                    'WEC: ava. available P from wind',
                                    'WEC: ava. available P technical reasons',
                                    'WEC: ava. Available P force majeure reasons',
                                    'WEC: ava. Available P force external reasons',
                                    'WEC: max. windspeed', 'WEC: min. windspeed', 
                                    'WEC: Operating Hours', 'WEC: Production kWh',
                                    'WEC: Production minutes', 'DateTime_y'])

train_df.dropna(inplace=True)

# Split the data into features (X) and target (y)
X = train_df.iloc[:,:-1]
y = train_df.iloc[:,-1]

# Preprocess non-numeric columns
X = pd.get_dummies(X)

def clean_column_names(df):
    df.columns = df.columns.str.replace('[^\w]', '_', regex=True)
    return df

X = clean_column_names(X)

# Encode string labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.4, random_state=42)

# Define the resampling and classification pipeline with SMOTE and Random Forest
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42))
])

# Set up the cross-validation with StratifiedKFold
cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

# Set up the scoring metrics
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, average='weighted'),
           'recall': make_scorer(recall_score, average='weighted'),
           'f1': make_scorer(f1_score, average='weighted')}

# Perform cross-validation
cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

# Print the cross-validation results
print("Mean Accuracy: {:.2f}".format(cv_results['test_accuracy'].mean()))
print("Mean Precision: {:.2f}".format(cv_results['test_precision'].mean()))
print("Mean Recall: {:.2f}".format(cv_results['test_recall'].mean()))
print("Mean F1 Score: {:.2f}".format(cv_results['test_f1'].mean()))

# Train the pipeline on the full training set
pipeline.fit(X_train, y_train)

# Test the pipeline on the test set
y_pred_encoded = pipeline.predict(X_test)

# Convert the integer predictions back to string labels
y_pred = le.inverse_transform(y_pred_encoded)

# Convert the integer true labels back to string labels
y_test_str = le.inverse_transform(y_test)

param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [3, 6, None],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__bootstrap': [True, False]
}

# Grid search CV
grid = GridSearchCV(pipeline, param_grid, verbose=1, cv=cv, n_jobs=-1, scoring='f1_macro')

# Fit grid on train set
grid.fit(X_train, y_train)

# Print the classification report
print(classification_report(y_test_str, y_pred))

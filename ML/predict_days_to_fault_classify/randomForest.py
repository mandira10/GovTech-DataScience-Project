import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load datasets
fault_data = pd.read_csv("/home/peverell/Documents/ds/selfmade_efe_mert_cem/data/fault_data.csv")
scada_data = pd.read_csv("/home/peverell/Documents/ds/selfmade_efe_mert_cem/data/scada_data.csv")

# Convert DateTime columns to pandas datetime type
scada_data['DateTime'] = pd.to_datetime(scada_data['DateTime'])
fault_data['DateTime'] = pd.to_datetime(fault_data['DateTime'])

# Merge datasets on 'Time' column, and create a copy to avoid SettingWithCopyWarning
df_combine = scada_data.merge(fault_data, on='Time', how='outer').copy()

# Replace NaN values in 'Fault' column with 'NF'
df_combine['Fault'] = df_combine['Fault'].replace(np.nan, 'NF')

# Sample 300 instances from 'NF' class
df_nf = df_combine[df_combine.Fault == 'NF'].sample(300, random_state=42)

# Sample 300 instances from fault class
df_fault = df_combine[df_combine.Fault != 'NF'].sample(300, random_state=42)

# Concatenate the sampled instances
df_combine = pd.concat([df_nf, df_fault], axis=0).reset_index(drop=True)

# Sort the DataFrame by 'DateTime_x' column
df_combine.sort_values('DateTime_x', inplace=True)

# Create a copy of DataFrame to avoid SettingWithCopyWarning
df = df_combine.copy() 

# Initialize 'TimeToNextFault' column with infinity
df['TimeToNextFault'] = np.inf

# Define the list of faults
faults = ['GF', 'AF', 'FF', 'MF', 'EF']

# Calculate time to next fault
next_fault_index = None
for i in reversed(df.index):
    if df.loc[i, 'Fault'] in faults:
        next_fault_index = i
    if next_fault_index is not None:
        df.loc[i, 'TimeToNextFault'] = (df.loc[next_fault_index, 'DateTime_x'] - df.loc[i, 'DateTime_x']).total_seconds() / 3600

# Set 'TimeToNextFault' to 0 for rows with faults
df.loc[df['Fault'].isin(faults), 'TimeToNextFault'] = 0

# Copy the DataFrame to avoid SettingWithCopyWarning
df_combine = df.copy() 

# Exclude rows where 'TimeToNextFault' is infinity
df_combine = df_combine[df_combine['TimeToNextFault'] != np.inf]

# Add 'DayOfYear' and 'Year' features
df_combine.loc[:, 'DayOfYear'] = df_combine['DateTime_x'].dt.dayofyear
df_combine.loc[:, 'Year'] = df_combine['DateTime_x'].dt.year

# Drop irrelevant columns
train_df = df_combine.drop(columns=['DateTime_x', 'DateTime_y','Fault'])

# Remove any rows with NaN values
train_df.dropna(inplace=True)

# Prepare features and target variable
X = train_df.drop(columns=['TimeToNextFault'])
threshold = 1
threshold_data = train_df['TimeToNextFault'] <= threshold
y = (threshold_data).astype(int)

# Convert categorical data into numerical
X = pd.get_dummies(X)

# Function to clean column names by replacing non-word characters
def clean_column_names(df):
    df.columns = df.columns.str.replace('[^\w]', '_', regex=True)
    return df

# Clean column names
X = clean_column_names(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipeline with StandardScaler and RandomForestClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Predict on testing set
y_pred = pipeline.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("F1-score:", f1)
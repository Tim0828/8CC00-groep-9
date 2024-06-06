# time to oversample the minority class data
import pandas as pd
import numpy as np

input_file = 'tested_molecules_with_descriptors.csv'
with open(input_file, 'r') as f:
    df = pd.read_csv(f)

# Identify columns with zero variance (have no discriminatory information)
zero_variance_columns = df.columns[df.nunique() <= 1]

# Print or remove these columns
print("Columns with zero variance:", zero_variance_columns)
df = df.drop(columns=zero_variance_columns)

# get the number of data points for each class
num_class_0 = df[(df['PKM2_inhibition'] == 0) & (df['ERK2_inhibition'] == 0)].shape[0]
num_class_1 = df[(df['PKM2_inhibition'] != 0) | (df['ERK2_inhibition'] != 0)].shape[0]

# get the data points for each class
class_0 = df[(df['PKM2_inhibition'] == 0) & (df['ERK2_inhibition'] == 0)]
class_1 = df[(df['PKM2_inhibition'] != 0) | (df['ERK2_inhibition'] != 0)]

# oversample the minority class
class_1_oversampled = class_1.sample(n=num_class_0, replace=True)

# combine the two classes
df_oversampled = pd.concat([class_0, class_1_oversampled])
print(df_oversampled.head())
print(df_oversampled.describe())

# save the data
output_file = 'oversampled_molecules_with_descriptors.csv'
with open(output_file, 'w') as f:
    df_oversampled.to_csv(f, index=False)
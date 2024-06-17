from iterative import FreshDeep
import pandas as pd
import numpy as np
# grab raw data
with open(r'deep learning\data\tested_molecules_with_descriptors.csv', 'r') as infile:
    df = pd.read_csv(infile)


# drop SMILES if not already
if 'SMILES' in df.columns:
    df = df.drop(columns = ['SMILES'])

# split X and y 
dfy = df[['PKM2_inhibition', 'ERK2_inhibition']]
df = df.drop(columns=['PKM2_inhibition', 'ERK2_inhibition'])
dfx = df

# Convert DataFrame to NumPy array
x = dfx.values
y = dfy.values

# fill nan values with 0
x = np.nan_to_num(x)
y = np.nan_to_num(y)

# test 'best model'
model = FreshDeep()
model.load(r'data\best_model.keras')
acc = model.evaluate(y_test=y,x_test=x)
bacc = model.balanced_accuracy(y_true=y,x_test=x)

print('acc: ', acc,' bacc: ', bacc)
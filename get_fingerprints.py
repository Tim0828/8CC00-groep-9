# get fingerprints from SMILES

from rdkit.Chem import PandasTools, AllChem
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split

def read_file(input_file):
    # file columns: "SMILES","PKM2_inhibition","ERK2_inhibition"
    with open(input_file, 'r') as infile:
        df = pd.read_csv(infile)
    return df
def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


input_file = 'data/tested_molecules.csv'
df = read_file(input_file)
df['features'] = df['SMILES'].apply(smiles_to_fingerprint)
df = df.dropna(subset=['features'])

# Convert list of arrays to 2D array
X_rf = np.array(df['features'].tolist())
y = df[['PKM2_inhibition', 'ERK2_inhibition']].values
print(X_rf)
print(y)

# Split the data into training and test sets
A_samples = df[df["PKM2_inhibition"] == 1]
B_samples = df[df["ERK2_inhibition"] == 1]
A_train, A_test = train_test_split(A_samples, test_size=0.2, random_state=42)
B_train, B_test = train_test_split(B_samples, test_size=0.2, random_state=42)
train = pd.concat([A_train, B_train])
test = pd.concat([A_test, B_test])
X_train = np.array(train['features'].tolist())
X_test = np.array(test['features'].tolist())
y_train = train[['PKM2_inhibition', 'ERK2_inhibition']].values
y_test = test[['PKM2_inhibition', 'ERK2_inhibition']].values

# save the data
np.save('data/X_train.npy', X_train)
np.save('data/X_test.npy', X_test)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)

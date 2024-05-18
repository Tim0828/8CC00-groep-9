import rdkit
import pandas as pd
# # Adding descriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools

with open('tested_molecules.csv', 'r') as infile:
    df = pd.read_csv(infile)

# add molecule column to the dataframe
PandasTools.AddMoleculeColumnToFrame(df, smilesCol='SMILES')

# Add all descriptors to the dataframe
for desc in Descriptors._descList:
    desc_name = desc[0]
    df[desc_name] = df['ROMol'].map(lambda x: desc[1](x))

# Save the dataframe
df.to_csv('tested_molecules_with_descriptors.csv', index=False)


import rdkit
import pandas as pd

with open('tested_molecules.csv', 'r') as infile:
    df = pd.read_csv(infile)


from rdkit.Chem import PandasTools
# add molecule column to the dataframe
PandasTools.AddMoleculeColumnToFrame(df, smilesCol='SMILES')

desc_list = [n[0] for n in Descriptors._descList]

# # Adding descriptors
from rdkit.Chem import Descriptors
# add each descriptor to the dataframe
for desc in desc_list:
    df[desc] = df['ROMol'].map(lambda x: Descriptors.descList[desc][0](x))

print(df.head())

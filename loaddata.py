import rdkit
import pandas as pd
# # Adding descriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools


def load_data(input_file):
    with open(input_file, 'r') as infile:
        df = pd.read_csv(infile)

    # add molecule column to the dataframe
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol='SMILES')

    # Add all descriptors to the dataframe
    for desc in Descriptors._descList:
        desc_name = desc[0]
        df[desc_name] = df['ROMol'].map(lambda x: desc[1](x))

    # drop ROmol for saving
    df.drop('ROMol', axis=1, inplace=True)
    # Save the dataframe
    input_file = input_file.split('.')[0]
    df.to_csv('{}_with_descriptors.csv'.format(input_file), index=False)

    return df

df = load_data('tested_molecules.csv')
print(df.head())
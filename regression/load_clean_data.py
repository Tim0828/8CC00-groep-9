import rdkit
import pandas as pd
# # Adding descriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools



def load_data(input_file):
   
    df = pd.read_csv(input_file)
    
    # Add molecule column to the dataframe
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol='SMILES')

    # Prepare a dictionary to hold descriptor values
    dict_Descriptors = {}

    # Add all descriptors to the dictionary
    for Descriptor_name, Descriptor_func in Descriptors._descList:
        dict_Descriptors[Descriptor_name] = df['ROMol'].map(lambda x: Descriptor_func(x))

    # Create a DataFrame using the dict
    descriptors_df = pd.DataFrame(dict_Descriptors)

    #add all values to the df in one step
    df = pd.concat([df, descriptors_df], axis=1)

    # Drop duplicates based on SMILES
    df = df.drop_duplicates(subset=['SMILES'])
    # Drop rows missing smiles
    df = df.dropna(subset=['SMILES'])
    # Drop ROMol column for saving
    df.drop('ROMol', axis=1, inplace=True)
    
    return df

input_file = 'tested_molecules.csv'
df = load_data(input_file)

#Now save the df
output_file = 'clean_{}.csv'.format(input_file.split('.')[0])
df.to_csv(output_file, index=False)
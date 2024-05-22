import rdkit
import pandas as pd
# # Adding descriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools

def load_data(input_file):
    with open(input_file, 'r') as infile:
        df = pd.read_csv(infile)

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

    # Drop ROMol column for saving
    df.drop('ROMol', axis=1, inplace=True)

    #Now save the df
    output_file = '{}_with_descriptors.csv'.format(input_file.split('.')[0])
    df.to_csv(output_file, index=False)

    return df

df = load_data('tested_molecules.csv')
print(df.head())
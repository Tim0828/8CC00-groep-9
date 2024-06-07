import pandas as pd
with open("tested_molecules_with_descriptors.csv", 'r') as f:
    df = pd.read_csv(f)

df_without_smiles = df.drop(columns = ["SMILES"])
df_without_smiles.to_csv("tested_molecules_without_SMILES.csv", index = False)




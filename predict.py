import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from load_clean_data import load_data



# Load your trained PCA model
pca_model = load_model('my_modelPCA999.keras')

df_untested_data = load_data("untested_molecules.csv")

df_descriptors = df_untested_data[['PKM2_inhibition', 'ERK2_inhibition']]
df_smiles = df_untested_data["SMILES"]

df_untested_data = df_untested_data.drop(columns=['PKM2_inhibition', 'ERK2_inhibition', 'SMILES'])


scalerminmax = MinMaxScaler()
df_scaled = scalerminmax.fit_transform(df_untested_data)
# Perform PCA
pca = PCA(n_components=43)
df_untested_mol_pca = pca.fit_transform(df_scaled)

# add the SMILES and inhibition labels back to the PCA data
df_untested_mol_pca = pd.DataFrame(df_untested_mol_pca, columns=[f'PC{i+1}' for i in range(43)])
df_untested_mol_pca = pd.concat([df_untested_mol_pca, df_descriptors], axis=1)


# Make predictions on the preprocessed untested dataset
predictions = pca_model.predict(df_untested_mol_pca)
prediction_labels = np.where(predictions >= 0.5, 1, 0)


df_untested_mol_pca[['PKM2_inhibition', 'ERK2_inhibition']] = prediction_labels
df_final = pd.DataFrame(columns=['SMILES','PKM2_inhibition', 'ERK2_inhibition'])

# You can populate 'df_final' with values from 'df_untested_mol_pca' as needed
df_final['SMILES'] = df_smiles
df_final['PKM2_inhibition'] = df_untested_mol_pca['PKM2_inhibition']
df_final['ERK2_inhibition'] = df_untested_mol_pca['ERK2_inhibition']
df_final.to_csv("prediction.csv", index=False)


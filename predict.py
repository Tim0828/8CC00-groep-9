import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from load_clean_data import load_data
import matplotlib.pyplot as plt



# Load your trained PCA model
pca_model = load_model('my_modelPCA999.keras')

if pca_model is None:
    print("Error: Failed to load the model.")
else:
    df_untested_data = load_data("untested_molecules.csv")

    df_descriptors = df_untested_data[['PKM2_inhibition', 'ERK2_inhibition']]
    df_smiles = df_untested_data["SMILES"]

    df_untested_data = df_untested_data.drop(columns=['PKM2_inhibition', 'ERK2_inhibition', 'SMILES'])


    scalerminmax = MinMaxScaler()
    df_scaled = scalerminmax.fit_transform(df_untested_data)
    # Perform PCA
    pca = PCA() #n_components=43
    df_untested_mol_pca = pca.fit_transform(df_scaled)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = explained_variance_ratio.cumsum()
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance)
    plt.show()
    # add the SMILES and inhibition labels back to the PCA data
    # df_untested_mol_pca = pd.DataFrame(df_untested_mol_pca, columns=[f'PC{i+1}' for i in range(80)])
    # df_untested_mol_pca = pd.concat([df_untested_mol_pca, df_descriptors], axis=1)

    # placeholder_value = -999  # Example placeholder value
    # df_untested_mol_pca_filled = df_untested_mol_pca.fillna(placeholder_value)
    

    # #Make predictions on the preprocessed data
    # predictions = pca_model.predict(df_untested_mol_pca_filled)
    # print(predictions)
    #Thresholding to convert probabilities into class labels
    # predicted_labels = (predictions >= 0.5).astype(int) 


    # df_untested_mol_pca[['PKM2_inhibition', 'ERK2_inhibition']] = predicted_labels
    # df_final = pd.DataFrame(columns=['SMILES','PKM2_inhibition', 'ERK2_inhibition'])

    # # You can populate 'df_final' with values from 'df_untested_mol_pca' as needed
    # df_final['SMILES'] = df_smiles
    # df_final['PKM2_inhibition'] = df_untested_mol_pca['PKM2_inhibition']
    # df_final['ERK2_inhibition'] = df_untested_mol_pca['ERK2_inhibition']
    # df_final.to_csv("prediction.csv", index=False)


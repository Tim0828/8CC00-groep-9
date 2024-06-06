from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
# Replace 'your_dataset.csv' with the actual filename
data = pd.read_csv('tested_molecules_with_descriptors.csv')

# Separate the molecular descriptors from the inhibition labels and SMILES
data_clean= data.drop(columns=['PKM2_inhibition', 'ERK2_inhibition', 'SMILES'])

# Standardize the features by scaling them to have mean 0 and variance 1
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_clean)


# Perform PCA to reduce the dimensionality of the data
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_scaled)

# Create a DataFrame to store the principal components and inhibition labels
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df[['PKM2_inhibition', 'ERK2_inhibition']] = data[['PKM2_inhibition', 'ERK2_inhibition']]
pca_df['SMILES'] = data[['SMILES']]
# Visualize the explained variance ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

scatter1 = axs[0].scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['PKM2_inhibition'], cmap='viridis')
axs[0].set_xlabel('PC1')
axs[0].set_ylabel('PC2')
axs[0].set_title('PCA: PC1 vs PC2 (Colored by PKM2 Inhibition)')
axs[0].grid(True)

# Plot for ERK2 inhibition
scatter2 = axs[1].scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['ERK2_inhibition'], cmap='viridis')
axs[1].set_xlabel('PC1')
axs[1].set_ylabel('PC2')
axs[1].set_title('PCA: PC1 vs PC2 (Colored by ERK2 Inhibition)')
axs[1].grid(True)

# Add colorbars to the plots
cbar1 = plt.colorbar(ax=axs[0], mappable=scatter1)
cbar1.set_label('PKM2 Inhibition')

cbar2 = plt.colorbar(ax=axs[1], mappable=scatter2)
cbar2.set_label('ERK2 Inhibition')

plt.tight_layout()
plt.show()
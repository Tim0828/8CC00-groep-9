import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# Load the dataset
file_path = "C:/Users/20201954/OneDrive - TU Eindhoven/2023/2023-4/8CC00 Advanced programming/tested_molecules_with_descriptors.csv"  # Update with the correct file path if needed
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Basic data exploration
#print(data.info())
#print(data.describe())

# Check for missing values
#print(data.isnull().sum())

# Visualize the distribution of each descriptor
#descriptors = data.columns[3:]  # Excluding SMILES and inhibition columns

# Histograms for all descriptors
#data[descriptors].hist(figsize=(20, 20), bins=30)
#plt.suptitle('Histograms of Molecular Descriptors')
#plt.show()


#corr_matrix = data[descriptors].corr()
#plt.figure(figsize=(15, 15))
#sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
#plt.title('Correlation Matrix of Molecular Descriptors')
#plt.show()


# Separate descriptors and targets
descriptors = data.columns[3:]  # Excluding SMILES and inhibition columns
targets = ['PKM2_inhibition', 'ERK2_inhibition']


# Descriptor-Descriptor Correlation
descriptor_corr_matrix = data[descriptors].corr()

# Heatmap of Descriptor-Descriptor Correlation Matrix
plt.figure(figsize=(15, 15))
sns.heatmap(descriptor_corr_matrix, cmap='coolwarm', annot=False)
plt.title('Descriptor-Descriptor Correlation Matrix')
plt.show()

# Descriptor-Inhibition Correlation
descriptor_inhibition_corr = data[descriptors].join(data[targets]).corr()

# Extracting Descriptor-Inhibition correlations
pkm2_correlations = descriptor_inhibition_corr[targets[0]].drop(targets)
erk2_correlations = descriptor_inhibition_corr[targets[1]].drop(targets)

# Sorting correlations
pkm2_correlations_sorted = pkm2_correlations.sort_values(ascending=False)
erk2_correlations_sorted = erk2_correlations.sort_values(ascending=False)

# Plotting Descriptor-PKM2 Inhibition Correlation
plt.figure(figsize=(10, 8))
pkm2_correlations_sorted.plot(kind='bar')
plt.title('Descriptor-PKM2 Inhibition Correlation')
plt.ylabel('Correlation Coefficient')
plt.xlabel('Descriptors')
plt.show()

# Plotting Descriptor-ERK2 Inhibition Correlation
plt.figure(figsize=(10, 8))
erk2_correlations_sorted.plot(kind='bar')
plt.title('Descriptor-ERK2 Inhibition Correlation')
plt.ylabel('Correlation Coefficient')
plt.xlabel('Descriptors')
plt.show()

# Display top 10 positively and negatively correlated descriptors for each target
top_pkm2_positive = pkm2_correlations_sorted.head(10)
top_pkm2_negative = pkm2_correlations_sorted.tail(10)
top_erk2_positive = erk2_correlations_sorted.head(10)
top_erk2_negative = erk2_correlations_sorted.tail(10)

print("Top 10 positively correlated descriptors with PKM2 inhibition:")
print(top_pkm2_positive)
#print("\nTop 10 negatively correlated descriptors with PKM2 inhibition:")
#print(top_pkm2_negative)
print("\nTop 10 positively correlated descriptors with ERK2 inhibition:")
print(top_erk2_positive)
#print("\nTop 10 negatively correlated descriptors with ERK2 inhibition:")
#print(top_erk2_negative)

# Plotting top 10 Descriptor-PKM2 Inhibition Correlation
plt.figure(figsize=(10, 8))
top_pkm2_positive.plot(kind='bar')
plt.title('Top 10 Descriptor-PKM2 Inhibition Correlation')
plt.ylabel('Correlation Coefficient')
plt.xlabel('Descriptors')
plt.show()

# Plotting top 10 Descriptor-ERK2 Inhibition Correlation
plt.figure(figsize=(10, 8))
top_erk2_positive.plot(kind='bar')
plt.title('Top 10 Descriptor-ERK2 Inhibition Correlation')
plt.ylabel('Correlation Coefficient')
plt.xlabel('Descriptors')
plt.show()
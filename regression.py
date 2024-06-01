import rdkit
import load_clean_data
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from load_clean_data import load_data
from sklearn.linear_model import LinearRegression
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#willen we ook een plot van de test en train data -> voor verschil met machine learning
#from sklearn.preprocessing import PolynomialFeatures 

#wachten tot EDA klaar is / wel code schrijven maar niet werkend nog
#beginnen met scatterplot om te zien hoe data er precies uitziet
#PCA (nog in te verdiepen in PCR (PCA regression))
#scale the data
#bepalen hoeveel parameters we meenemen
#zien welke parameters meest significant is
#bepalen hoeveel procent 90-95%

#voedingsstoffen= parameters 
#fruit/groenten = de inhibitoren

df_untested_data = load_data("untested_molecules.csv")

df_descriptors = df_untested_data[['PKM2_inhibition', 'ERK2_inhibition']]
df_smiles = df_untested_data["SMILES"]

df_untested_data = df_untested_data.drop(columns=['PKM2_inhibition', 'ERK2_inhibition', 'SMILES'])


scalerminmax = MinMaxScaler()
df_scaled = scalerminmax.fit_transform(df_untested_data)


pca = PCA() 
pca.fit_transform(df_scaled)
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
# Determine the number of components needed to explain 90% of the variance
desired_variance = 0.90
num_components = np.argmax(cumulative_explained_variance >= desired_variance) + 1
#loose components who have too little influence
pca=PCA(n=num_components)
df_untested_mol_pca = pca.fit_transform(df_scaled)


X_train, X_test, y_train, y_test = train_test_split(df_untested_mol_pca,df_descriptors, test_size=0.2, random_state=42)

# Fit the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

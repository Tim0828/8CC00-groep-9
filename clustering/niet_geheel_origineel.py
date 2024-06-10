import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, LSTM, Dense
# from sklearn.ensemble import VotingClassifier
# from sklearn.metrics import precision_recall_fscore_support
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

# Load data (assuming it's stored in a CSV file)
data = pd.read_csv(r"C:\Users\20223192\OneDrive - TU Eindhoven\Documents\Advanced programming\8CC00-groep-9\data\tested_molecules.csv")
data.columns = ["SMILES", "PKM2_inhibition", "ERK2_inhibition"]

# Convert inhibition columns to integers
data["PKM2_inhibition"] = data["PKM2_inhibition"].astype(int)
data["ERK2_inhibition"] = data["ERK2_inhibition"].astype(int)

# Preprocessing for Random Forest
# Convert SMILES strings to numerical features (e.g., molecular fingerprints)
# Function to convert SMILES to Morgan fingerprints
def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

# Apply the function to create the feature matrix
data['features'] = data['SMILES'].apply(smiles_to_fingerprint)
data = data.dropna(subset=['features'])

# Convert list of arrays to 2D array
X_rf = np.array(data['features'].tolist())

# Create the y matrix for multi-label classification
y = data[['PKM2_inhibition', 'ERK2_inhibition']].values

# Split the data into training and test sets
A_samples = data[data["PKM2_inhibition"] == 1]
B_samples = data[data["ERK2_inhibition"] == 1]
A_train, A_test = train_test_split(A_samples, test_size=0.2, random_state=42)
B_train, B_test = train_test_split(B_samples, test_size=0.2, random_state=42)
train_data_rf = pd.concat([A_train, B_train], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
test_data_rf = pd.concat([A_test, B_test], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and labels for Random Forest
X_train_rf, y_train_rf = np.array(train_data_rf['features'].tolist()), train_data_rf[['PKM2_inhibition', 'ERK2_inhibition']].values
X_test_rf, y_test_rf = np.array(test_data_rf['features'].tolist()), test_data_rf[['PKM2_inhibition', 'ERK2_inhibition']].values

# Train and Evaluate the Random Forest Model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_rf, y_train_rf)

rf_predictions = rf_classifier.predict(X_test_rf)
rf_report = classification_report(y_test_rf, rf_predictions, target_names=["PKM2_inhibition", "ERK2_inhibition"], zero_division=0)

print("Random Forest Model:")
print(rf_report)
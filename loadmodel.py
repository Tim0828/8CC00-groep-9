# imports
from keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# VAL_SPLIT = 0.02

input_file = 'oversampled_molecules_with_descriptors.csv'
# training data 
with open(input_file, 'r') as infile:
        df = pd.read_csv(infile)

# split X and y 
dfy = df[['PKM2_inhibition', 'ERK2_inhibition']]
df = df.drop(columns=['PKM2_inhibition', 'ERK2_inhibition', 'SMILES'])
dfx = df

# Convert DataFrame to NumPy array
x = dfx.values
y = dfy.values


# load the model
model_name = 'my_model.keras'
model = load_model(model_name)

# # Evaluate the model on the test data using `evaluate`
# print("Evaluate on test data")
# results = model.evaluate(x, y, batch_size=64)
# print("test loss, test acc:", results)

# Generate predictions
# validate 
y_pred = model.predict(x)
y_pred_binary = np.where(y_pred > 0.5, 1, 0)
# calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y, y_pred_binary)
print("Accuracy over test: %.2f%%" % (accuracy*100))

# df_predictions = pd.DataFrame(y, columns=['PKM2_inhibition', 'ERK2_inhibition'])
# print(df_predictions.head())
# df_predictions.to_csv('predictions1.csv', index=False)


# # Plot the results
# import matplotlib.pyplot as plt
# plt.scatter(y, predictions)
# plt.xlabel('True Values')
# plt.ylabel('Predictions')
# plt.show()
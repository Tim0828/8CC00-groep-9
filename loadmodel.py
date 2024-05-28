# imports
# from keras.models import load_model
import numpy as np
import keras
from sklearn.model_selection import train_test_split
import pandas as pd


# load the model
model_name = 'my_model.keras'
model = keras.models.load_model(model_name)


# model.summary() #ignore this attribute error

# load the data

with open('tested_molecules_with_descriptors.csv') as infile:
    df = pd.read_csv(infile)

# Choose X and y
dfx = df.iloc[:, 3:]
dfy = df.iloc[:, 1:3]

# Convert DataFrame to NumPy array
x = dfx.values
y = dfy.values


# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x, y, batch_size=64)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(x)
print("predictions shape:", predictions.shape)
df_predict = pd.DataFrame(predictions)
df_predict.head()
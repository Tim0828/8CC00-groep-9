# imports
# from keras.models import load_model
import numpy as np
import keras
from sklearn.model_selection import train_test_split
import pandas as pd

VAL_SPLIT = 0.25

from sklearn.model_selection import train_test_split

input_file = 'PCA_data.csv'
# training data 
with open(input_file, 'r') as infile:
        df = pd.read_csv(infile)

# split X and y 
dfy = df[['PKM2_inhibition', 'ERK2_inhibition']]
df.drop(columns=['PKM2_inhibition', 'ERK2_inhibition'])
dfx = df

# Convert DataFrame to NumPy array
x = dfx.values
y = dfy.values

# load the model
model_name = 'my_modelPCA955.keras'
model = keras.models.load_model(model_name)




# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x, y, batch_size=64)
print("test loss, test acc:", results)

# # Generate predictions (probabilities -- the output of the last layer)
# # on new data using `predict`
# print("Generate predictions for 3 samples")
# predictions = model.predict(x)
# print("predictions shape:", predictions.shape)
# df_predict = pd.DataFrame(predictions)
# df_predict.head()
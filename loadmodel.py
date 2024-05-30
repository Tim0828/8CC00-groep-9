# imports
from keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# VAL_SPLIT = 0.02

input_file = 'PCA_data.csv'
# training data 
with open(input_file, 'r') as infile:
        df = pd.read_csv(infile)

# split X and y 
dfy = df[['PKM2_inhibition', 'ERK2_inhibition']]
df = df.drop(columns=['PKM2_inhibition', 'ERK2_inhibition'])
dfx = df

# Convert DataFrame to NumPy array
x = dfx.values
print(x.shape)
y = dfy.values

# # get only the last ..% of the data
# x = x[-int(len(x)*VAL_SPLIT):]
# y = y[-int(len(y)*VAL_SPLIT):]

# load the model
model_name = 'my_modelPCA1.keras'
model = load_model(model_name)

# # Evaluate the model on the test data using `evaluate`
# print("Evaluate on test data")
# results = model.evaluate(x, y, batch_size=64)
# print("test loss, test acc:", results)

# Generate predictions
y = model.predict(x)
df_predictions = pd.DataFrame(y, columns=['PKM2_inhibition', 'ERK2_inhibition'])
print(df_predictions.head())
df_predictions.to_csv('predictions1.csv', index=False)


# # Plot the results
# import matplotlib.pyplot as plt
# plt.scatter(y, predictions)
# plt.xlabel('True Values')
# plt.ylabel('Predictions')
# plt.show()
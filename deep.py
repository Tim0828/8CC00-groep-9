import numpy as np
import keras
from keras import layers


# # input
# inputs = keras.Input(shape=(210,))

# # new node in the graph of layers
# dense = layers.Dense(64, activation="relu")
# x = dense(inputs)

# # another node
# x = layers.Dense(64, activation="relu")(x)
# outputs = layers.Dense(2)(x)

# # model
# model = keras.Model(inputs=inputs, outputs=outputs, name="model")

# # summary
# model.summary()

# training data 
import pandas as pd
with open("tested_molecules_with_descriptors.csv", 'r') as infile:
        df = pd.read_csv(infile)

dfx = df.iloc[:, 3:]
dfy = df.iloc[:, 1:3]

# Convert DataFrame to NumPy array
x = dfx.values
y = dfy.values

from sklearn.model_selection import train_test_split

# Split the data into training and test sets    
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape, y_train.shape)

from keras.layers import Dropout, BatchNormalization, Embedding
from keras.callbacks import EarlyStopping

model = keras.Sequential(
    [
        keras.Input(shape=(210,)),
        layers.Dense(128, activation="relu"),
        Dropout(0.3),
        layers.Dense(128, activation="relu"),
        Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(2, activation="softmax"),
    ]
)

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()
# add early stopping (prevents overfitting)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(x_train, y_train, batch_size=64, epochs=30, validation_split=0.2, callbacks=[early_stopping])

test_scores = model.evaluate(x_test, y_test)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
model.save("my_model.keras")

# model.compile(
#     loss=keras.losses.CategoricalCrossentropy(from_logits=True),
#     optimizer=keras.optimizers.RMSprop(),
#     metrics=["accuracy"],
# )

# history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

# test_scores = model.evaluate(x_test, y_test)
# print("Test loss:", test_scores[0])
# print("Test accuracy:", test_scores[1])
# model.save("my_model.keras")
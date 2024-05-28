import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt
import pandas as pd

BATCH_SIZE = 64
EPOCHS = 300

def get_data(input_file):
    from sklearn.model_selection import train_test_split

    # training data 
    with open("clean_tested_molecules.csv", 'r') as infile:
            df = pd.read_csv(infile)

    # split X and y
    dfx = df.iloc[:, 3:]
    dfy = df.iloc[:, 1:3]

    # Convert DataFrame to NumPy array
    x = dfx.values
    y = dfy.values

    # Split the data into training and test sets    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = get_data("clean_tested_molecules.csv")

def get_model():
    from keras.layers import Dropout, Normalization #, BatchNormalization, Embedding

    model = keras.Sequential(
        [
            keras.Input(shape=(210,)),
            Normalization(),
            layers.Dense(256, activation="relu"),
            Dropout(0.2),
            layers.Dense(128, activation="relu"),
            Dropout(0.1),
            layers.Dense(64, activation="relu"),
            layers.Dense(2, activation="softmax"),
        ]
    )

    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

model = get_model()

from keras.callbacks import EarlyStopping

# model.summary()
# add early stopping (prevents overfitting)
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, callbacks=[early_stopping])

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

test_scores = model.evaluate(x_test, y_test)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
# model.save("my_model.keras")
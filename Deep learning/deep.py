import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt
import pandas as pd

BATCH_SIZE = 128
EPOCHS = int(11000/BATCH_SIZE)
VAL_SPLIT = 0.2

def get_data(input_file, VAL_SPLIT):
    from sklearn.model_selection import train_test_split

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


    # Split the data into training and test sets    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=VAL_SPLIT, random_state=42)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = get_data("oversampled_molecules_with_descriptors.csv", VAL_SPLIT)

def get_model(input_size):
    from keras.layers import Dropout #, Normalization #, BatchNormalization, Embedding

    model = keras.Sequential(
        [
            keras.Input(shape=(input_size,)),
            layers.Dense(256, activation="relu"),
            Dropout(0.2), # to prevent overfitting
            layers.Dense(128, activation="relu"),
            Dropout(0.2),
            layers.Dense(64, activation="relu"),
            Dropout(0.2),
            layers.Dense(32, activation="relu"),
            Dropout(0.2),
            layers.Dense(16, activation="relu"), # model seems to be performing better with only relu instead of leaky relu
            Dropout(0.2),
            layers.Dense(8, activation="relu"),
            Dropout(0.2),
            layers.Dense(4, activation="relu"), # converging shape seems to perform best
            Dropout(0.2),
            layers.Dense(2, activation="sigmoid"), #softmax would only allow for one 1 at a time
        ]
    )

    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

input_size = x_train.shape[1]
model = get_model(input_size)

def train_and_validate(model, VAL_SPLIT):
    from keras.callbacks import EarlyStopping

    # model.summary()
    # add early stopping (prevents overfitting)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VAL_SPLIT, callbacks=[early_stopping])

    # # list all data in history
    # print(history.history.keys())
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
    return

train_and_validate(model, VAL_SPLIT)

# validate 
y_pred = model.predict(x_test)
y_pred_binary = np.where(y_pred > 0.5, 1, 0)
# calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_binary)
print("Binary Accuracy over X_test: %.2f%%" % (accuracy*100))
# calculate precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred_binary, average='weighted')
print("Precision over X_test: %.2f%%" % (precision*100))


# save the model
#model.save("my_model.keras")
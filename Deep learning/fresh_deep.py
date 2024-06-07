# ahh a fresh start

# imports
import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns

# constants
VAL_SPLIT = 0.2

# load the data
def get_data(input_file, VAL_SPLIT):
    # training data 
    with open(input_file, 'r') as infile:
            df = pd.read_csv(infile)

    # apply SMOTE to balance the data

    # Convert all column names to strings
    df.columns = df.columns.astype(str)

    # apply smote to balance the data
    smote = SMOTE(sampling_strategy='minority')

    resampled_data_1 = smote.fit_resample(df.drop(columns=['ERK2_inhibition']), df['PKM2_inhibition'])
    resampled_data_2 = smote.fit_resample(df.drop(columns=['PKM2_inhibition']), df['ERK2_inhibition'])

    # Convert the resampled data to a DataFrame
    df1 = pd.DataFrame(data=resampled_data_1[0], columns=df.drop(columns=['ERK2_inhibition']).columns)
    df1['PKM2_inhibition'] = resampled_data_1[1]

    df2 = pd.DataFrame(data=resampled_data_2[0], columns=df.drop(columns=['PKM2_inhibition']).columns)
    df2['ERK2_inhibition'] = resampled_data_2[1]

    # Concatenate the two DataFrames
    df = pd.concat([df1, df2], axis=0)
    
    # split X and y 
    dfy = df[['PKM2_inhibition', 'ERK2_inhibition']]
    df = df.drop(columns=['PKM2_inhibition', 'ERK2_inhibition'])
    dfx = df

    # Convert DataFrame to NumPy array
    x = dfx.values
    y = dfy.values

    # fill nan values with 0
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)

    

    # # print minority class distribution
    # print('Minority class distribution: ', np.sum(y, axis=0))
    # # print minority class ratio
    # print('Minority class ratio: ', np.sum(y, axis=0) / y.shape[0])


    # Split the data into training and test sets    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=VAL_SPLIT, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # # plot x distribution
    # plt.hist(x_test.flatten(), bins=1000)
    # plt.title('x distribution')
    # plt.show()
    return x_train, x_test, y_train, y_test

class FreshDeep:
      
    def __init__(self) -> None:
        self.model = self.build_model()
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
        self.history = None
    def build_model(self):
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
            Dropout(0.25),
            layers.Dense(64, activation='relu'),
            Dropout(0.25),
            layers.Dense(32, activation='relu'),
            Dropout(0.25),
            layers.Dense(2, activation='sigmoid')
        ])
        return model
    def train(self, x_train, y_train, epochs=100):
        self.history = self.model.fit(x_train, y_train, epochs=epochs, validation_split=0.2, callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    def plot(self):
        plt.plot(self.history.history['auc'])
        plt.plot(self.history.history['val_auc'])
        plt.title('model auc')
        plt.ylabel('auc')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)
    def predict(self, x):
        return self.model.predict(x)
    def save(self, filename):
        self.model.save(filename)
    def load(self, filename):
        self.model = keras.models.load_model(filename)

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

# get the data
x_train, x_test, y_train, y_test = get_data("tested_molecules_without_SMILES.csv", VAL_SPLIT)

# train the model
model = FreshDeep()
model.train(x_train, y_train)

# plot the model
model.plot()

# evaluate the model
result = model.evaluate(x_test, y_test)
print(result)


# print confusion matrix
y_pred = model.predict(x_test)
y_pred = np.round(y_pred)
# print('Confusion matrix:')
# print(np.sum(y_test, axis=0))
# print(np.sum(y_pred, axis=0))

for i, target in enumerate(['PKM2_inhibition', 'ERK2_inhibition']):
    print(f'Confusion matrix for {target}:')
    plot_confusion_matrix(y_test[:, i], y_pred[:, i], title=f'Confusion Matrix for {target}')
# save the model
#model.save('fresh_model.keras')


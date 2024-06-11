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
from tensorflow.keras import backend as K
# constants
VAL_SPLIT = 0.2
def apply_SMOTE(df):
     # apply SMOTE to balance the data    
    # # Convert all column names to strings
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
    return df

def get_SMOTEdata(input_file, VAL_SPLIT):
    # training data 
    with open(input_file, 'r') as infile:
            df = pd.read_csv(infile)


    # drop SMILES if not already
    if 'SMILES' in df.columns:
        df = df.drop(columns = ['SMILES'])

    df = apply_SMOTE(df)
    
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

    # Split the data into training and test sets    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=VAL_SPLIT, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test

def get_data(input_file, VAL_SPLIT):
    # training data 
    with open(input_file, 'r') as infile:
            df = pd.read_csv(infile)


    # drop SMILES if not already
    if 'SMILES' in df.columns:
        df = df.drop(columns = ['SMILES'])
    
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

    # Split the data into training and test sets    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=VAL_SPLIT, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test

class FreshDeep:
      
    def __init__(self) -> None:
        self.model = self.build_model()
        self.model.optimizer = keras.optimizers.Adam(learning_rate=0.01)
        self.optimizer = self.model.optimizer
        self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['AUC'])
        self.history = None
    def build_model(self):
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
            Dropout(0.25),
            layers.Dense(64, activation='relu'),
            Dropout(0.25),
            layers.Dense(32, activation='relu'),
            Dropout(0.25),
            layers.Dense(16, activation='relu'),
            Dropout(0.25),
            layers.Dense(2, activation='sigmoid')
        ])
        return model
    def train(self, x_train, y_train, epochs=1):
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
    def conf_matrix(self, y_true, x_test):
        y_pred = self.model.predict(x_test)
        y_pred = np.round(y_pred)
        cm_pkm = confusion_matrix(y_true[:, 0], y_pred[:, 0])
        cm_erk = confusion_matrix(y_true[:, 1], y_pred[:, 1])
        return cm_erk, cm_pkm
    def balanced_accuracy(self, y_true, x_test):
        cm_erk, cm_pkm = self.conf_matrix(y_true, x_test)
        tn, fp, fn, tp = cm_erk.ravel()
        erk_ba = (tp / (tp + fn) + tn / (tn + fp)) / 2
        tn, fp, fn, tp = cm_pkm.ravel()
        pkm_ba = (tp / (tp + fn) + tn / (tn + fp)) / 2
        return erk_ba, pkm_ba

# def plot_confusion_matrix(model, y_true, title):
#     cm = model.conf_matrix(y_true)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title(title)
#     plt.show()


# get the data
x_train, x_test, y_train, y_test = get_SMOTEdata(r"data\tested_molecules_with_descriptors.csv", VAL_SPLIT)

# train the model
model = FreshDeep()
def iterative_training(model, iterations, initial_lr, x_train, x_test, y_train, y_test):
    i = 0
    log = [(0, 0)]
    while i < iterations:
        # Adjust learning rate
        new_lr = initial_lr / (i + 1)
        K.set_value(model.optimizer.learning_rate, new_lr)

        model.train(x_train, y_train)
        current = tuple(model.balanced_accuracy(y_test, x_test))
        log.append(current)
        if len(log) > 1:
            if log[-1] > log[-2]:
                best = log[-1]
                print(f'New best balanced accuracies: {best}')
                model.save(r'data\best_model.keras')
        i += 1
    return model
        
model = iterative_training(model, 1, 0.01, x_train, x_test, y_train, y_test)

# # evaluate the model on original data
# x_train, x_test, y_train, y_test = get_data(r"data\tested_molecules_with_descriptors.csv", VAL_SPLIT)
# print(model.balanced_accuracy(y_test, x_test))
# # evaluate the model
# result = model.evaluate(x_test, y_test)
# print(result)

# # print confusion matrix
# y_pred = model.predict(x_test)
# y_pred = np.round(y_pred)

# for i, target in enumerate(['PKM2_inhibition', 'ERK2_inhibition']):
#     # print_confusion_matrix(y_test[:, i], y_pred[:, i], target)
#     plot_confusion_matrix(y_test[:, i], y_pred[:, i], title=f'Confusion Matrix for {target}')
# # save the model
# # model.save('fresh_model.keras')


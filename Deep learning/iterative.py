import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras import backend as K
import logging
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
VAL_SPLIT = 0.2
BEST_MODEL_PATH = 'data/best_model.keras'
DATA_PATH = 'data/PCA_data.csv'
TEST_DATA_PATH = 'deep learning/data/tested_molecules_with_descriptors.csv'

def apply_SMOTE(df):
    # Apply SMOTE to balance the data    
    df.columns = df.columns.astype(str)

    smote = SMOTE(sampling_strategy='minority')

    resampled_data_1 = smote.fit_resample(df.drop(columns=['ERK2_inhibition']), df['PKM2_inhibition'])
    resampled_data_2 = smote.fit_resample(df.drop(columns=['PKM2_inhibition']), df['ERK2_inhibition'])

    df1 = pd.DataFrame(data=resampled_data_1[0], columns=df.drop(columns=['ERK2_inhibition']).columns)
    df1['PKM2_inhibition'] = resampled_data_1[1]

    df2 = pd.DataFrame(data=resampled_data_2[0], columns=df.drop(columns=['PKM2_inhibition']).columns)
    df2['ERK2_inhibition'] = resampled_data_2[1]

    df = pd.concat([df1, df2], axis=0)
    return df

def get_data(input_file, val_split):
    df = pd.read_csv(input_file)

    if 'SMILES' in df.columns:
        df = df.drop(columns=['SMILES'])
    
    y_df = df[['PKM2_inhibition', 'ERK2_inhibition']]
    x_df = df.drop(columns=['PKM2_inhibition', 'ERK2_inhibition'])

    x = x_df.values
    y = y_df.values

    x = np.nan_to_num(x)
    y = np.nan_to_num(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=val_split, random_state=42)

    df_train = np.concatenate((x_train, y_train), axis=1)
    df_train = pd.DataFrame(df_train, columns=df.columns.to_list() + y_df.columns.to_list())

    df_train = apply_SMOTE(df_train)

    y_train = df_train[['PKM2_inhibition', 'ERK2_inhibition']].values
    x_train = df_train.drop(columns=['PKM2_inhibition', 'ERK2_inhibition']).values
    
    return x_train, x_test, y_train, y_test

class FreshDeep:
    def __init__(self) -> None:
        self.model = self.build_model()
        self.model.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.optimizer = self.model.optimizer
        self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.history = None

    def build_model(self):
        model = tf.keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(x_train.shape[1],)),
            Dropout(0.25),
            layers.Dense(128, activation='relu'),
            Dropout(0.25),
            layers.Dense(64, activation='relu'),
            Dropout(0.25),
            layers.Dense(32, activation='relu'),
            Dropout(0.25),
            layers.Dense(16, activation='relu'),
            Dropout(0.25),
            layers.Dense(8, activation='relu'),
            Dropout(0.25),
            layers.Dense(4, activation='relu'),
            Dropout(0.25),
            layers.Dense(2, activation='sigmoid')
        ])
        return model

    def train(self, x_train, y_train, epochs=1, w=1):
        self.history = self.model.fit(x_train, y_train, epochs=epochs, validation_split=VAL_SPLIT, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)], class_weight={0: 1, 1: w})

    def plot(self):
        plt.plot(self.history.history['accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
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
        self.model = tf.keras.models.load_model(filename)

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

def build_keras_model():
    model = tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(x_train.shape[1],)),
        Dropout(0.25),
        layers.Dense(128, activation='relu'),
        Dropout(0.25),
        layers.Dense(64, activation='relu'),
        Dropout(0.25),
        layers.Dense(32, activation='relu'),
        Dropout(0.25),
        layers.Dense(16, activation='relu'),
        Dropout(0.25),
        layers.Dense(8, activation='relu'),
        Dropout(0.25),
        layers.Dense(4, activation='relu'),
        Dropout(0.25),
        layers.Dense(2, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def perform_hyperparameter_tuning(x_train, y_train):
    model = KerasClassifier(build_fn=build_keras_model, verbose=0)
    param_grid = {
        'batch_size': [10, 20, 40],
        'epochs': [10, 20, 30]
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(x_train, y_train)
    logging.info(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
    return grid_result.best_estimator_

def iterative_training(model, iterations, initial_lr, x_train, x_test, y_train, y_test):
    i = 0
    log = [(0, 0)]
    best_balanced_accuracy = (0, 0)

    while i < iterations:
        logging.info(f'Iteration {i+1}')
        w = 1 * 10**i
        model.train(x_train, y_train, epochs=25, w=w)
        current = tuple(model.balanced_accuracy(y_test, x_test))
        log.append(current)
        if current > best_balanced_accuracy:
            best_balanced_accuracy = current
            logging.info(f'New best balanced accuracies: {best_balanced_accuracy}')
            model.save(BEST_MODEL_PATH)
        i += 1
    return log

# Get the data
x_train, x_test, y_train, y_test = get_data(DATA_PATH, VAL_SPLIT)

# Hyperparameter tuning
best_model = perform_hyperparameter_tuning(x_train, y_train)

# Train the model
model = FreshDeep()
log = iterative_training(model, 10, 0.01, x_train, x_test, y_train, y_test)
logging.info(log)

# Test the best model
model.load(BEST_MODEL_PATH)
df = pd.read_csv(TEST_DATA_PATH)

if 'SMILES' in df.columns:
    df = df.drop(columns=['SMILES'])

y_df = df[['PKM2_inhibition', 'ERK2_inhibition']]
x_df = df.drop(columns=['PKM2_inhibition', 'ERK2_inhibition'])

x = x_df.values
y = y_df.values

x = np.nan_to_num(x)
y = np.nan_to_num(y)

cm1, cm2 = model.conf_matrix(y, x)
acc = model.evaluate(x_test=x, y_test=y)
bacc = model.balanced_accuracy(y_true=y, x_test=x)

logging.info(f'acc: {acc} bacc: {bacc}')
logging.info(cm1)
logging.info(cm2)

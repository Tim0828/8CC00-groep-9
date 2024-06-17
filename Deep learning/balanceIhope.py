from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

class BalancedSparseCategoricalAccuracy(keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, name='balanced_sparse_categorical_accuracy', dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_flat = y_true
        if y_true.shape.ndims == y_pred.shape.ndims:
            y_flat = tf.squeeze(y_flat, axis=[-1])
        y_true_int = tf.cast(y_flat, tf.int32)

        cls_counts = tf.math.bincount(y_true_int)
        cls_counts = tf.math.reciprocal_no_nan(tf.cast(cls_counts, self.dtype))
        weight = tf.gather(cls_counts, y_true_int)
        return super().update_state(y_true, y_pred, sample_weight=weight)

class BalancedClassificationModel:
    def __init__(self, input_size):
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=input_size, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[BalancedSparseCategoricalAccuracy()])

    def train(self, X_train, y_train, epochs=20, batch_size=64):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, class_weight={0: 1., 1: 141})

    def evaluate(self, X_test, y_test):
        # confusion matrix
        y_pred = (self.model.predict(X_test) > 0.5).astype("int32")
        cm = confusion_matrix(y_test, y_pred)
        bacc = (cm[0, 0] / (cm[0, 0] + cm[0, 1]) + cm[1, 1] / (cm[1, 0] + cm[1, 1])) / 2
        return bacc, cm
    

# input data
df = pd.read_csv('data/normal_tested.csv')
# X_ERK = pd.read_csv('data/X_best_ERK2_pca.csv')

# df = pd.read_csv('data/tested_molecules.csv')
# extract PKM2 and ERK2 inhibition values
y_PKM = df['PKM2_inhibition']
y_ERK = df['ERK2_inhibition']
X = df.drop(columns=['PKM2_inhibition', 'ERK2_inhibition'])

# I am paranoid
# check if 'PKM2_inhibition', 'ERK2_inhibition'in columns
if 'PKM2_inhibition' in X.columns:
    X = X.drop(columns=['PKM2_inhibition'])
    print('PKM2_inhibition dropped')
if 'ERK2_inhibition' in X.columns:
    X = X.drop(columns=['ERK2_inhibition'])
    print('ERK2_inhibition dropped')
# y_PKM = df['PKM2_inhibition'][:-1] # drop last row (X had one less row)
# y_ERK = df['ERK2_inhibition'][:-1]

input_size = X.shape[1]
balanced_model = BalancedClassificationModel(input_size=input_size)
balanced_model.train(X, y_ERK)
bacc, cm = balanced_model.evaluate(X, y_ERK)
print(f"Balanced accuracy ERK: {bacc}")
print(f"Confusion matrix ERK:\n{cm}")
balanced_model.train(X, y_PKM)
bacc, cm = balanced_model.evaluate(X, y_PKM)
print(f"Balanced accuracy PKM: {bacc}")
print(f"Confusion matrix PKM:\n{cm}")
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


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
        self.model.add(Dense(128, input_dim=input_size, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[BalancedSparseCategoricalAccuracy()])

    def train(self, X_train, y_train, epochs=20, batch_size=64, w=140):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, class_weight={0: 1., 1: w})

    def evaluate(self, X_test, y_test):
        # confusion matrix
        y_pred = (self.model.predict(X_test) > 0.5).astype("int32")
        cm = confusion_matrix(y_test, y_pred)
        bacc = (cm[0, 0] / (cm[0, 0] + cm[0, 1]) + cm[1, 1] / (cm[1, 0] + cm[1, 1])) / 2
        return bacc, cm
    

# # input data
X_PKM2_pca = pd.read_csv('data\X_PKM2_pca.csv')
y_PKM2_pca = pd.read_csv('data\y_PKM2.csv')
X_ERK2_pca = pd.read_csv('data\X_ERK2_pca.csv')
y_ERK2_pca = pd.read_csv('data\y_ERK2.csv')
# df = pd.read_csv('data/clean_tested_molecules.csv')
# # X_ERK = pd.read_csv('data/X_best_ERK2_pca.csv')

# # df = pd.read_csv('data/tested_molecules.csv')
# # extract PKM2 and ERK2 inhibition values
# y_PKM = df['PKM2_inhibition']
# y_ERK = df['ERK2_inhibition']
# X = df.drop(columns=['PKM2_inhibition', 'ERK2_inhibition'])

# I am paranoid
# check if 'PKM2_inhibition', 'ERK2_inhibition'in columns
# if 'PKM2_inhibition' in X.columns:
#     X = X.drop(columns=['PKM2_inhibition'])
#     print('PKM2_inhibition dropped')
# if 'ERK2_inhibition' in X.columns:
#     X = X.drop(columns=['ERK2_inhibition'])
#     print('ERK2_inhibition dropped')
# y_PKM = df['PKM2_inhibition'][:-1] # drop last row (X had one less row)
# y_ERK = df['ERK2_inhibition'][:-1]

# split data
X_train, X_test, y_ERK_train, y_ERK_test = train_test_split(X_ERK2_pca, y_ERK2_pca, test_size=0.25)
X_train2, X_test2, y_PKM_train, y_PKM_test = train_test_split(X_PKM2_pca, y_PKM2_pca, test_size=0.25)

input_size = X_train2.shape[1]
balanced_model = BalancedClassificationModel(input_size=input_size)

weights_PKM = list(range(80, 120, 2))
# PKM
baccs2 = []
for weight in weights_PKM:
    balanced_model.train(X_train2, y_PKM_train, epochs=15, w=weight)
    bacc, cm = balanced_model.evaluate(X_test2, y_PKM_test)
    baccs2.append({'weight': weight, 'bacc': bacc, 'cm': cm})

# ERK
weights_ERK = list(range(40, 80, 2))
baccs = []
for weight in weights_ERK:
    balanced_model.train(X_train, y_ERK_train, epochs=15, w=weight)
    bacc, cm = balanced_model.evaluate(X_test, y_ERK_test)
    baccs.append({'weight': weight, 'bacc': bacc, 'cm': cm})

# print highest bacc
best2 = max(baccs2, key=lambda x: x['bacc'])
print(f"Best balanced accuracy PKM: {best2['bacc']} with weight {best2['weight']}")
print(f"Confusion matrix: {best2['cm']}")

# print highest bacc
best = max(baccs, key=lambda x: x['bacc'])
print(f"Best balanced accuracy ERK: {best['bacc']} with weight {best['weight']}")
print(f"Confusion matrix: {best['cm']}")
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.metrics import Metric
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.framework import dtypes

# class BalancedBinaryAccuracy(Metric):
#     def __init__(self, name='balanced_binary_accuracy', dtype=None):
#         super(BalancedBinaryAccuracy, self).__init__(name, dtype=dtype)
#         self.total = self.add_weight('total', initializer=init_ops.zeros_initializer)
#         self.count = self.add_weight('count', initializer=init_ops.zeros_initializer)

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true = math_ops.cast(y_true, dtypes.float32)
#         y_pred = math_ops.cast(y_pred, dtypes.float32)
        
#         # Compute the number of positive and negative samples
#         pos_count = math_ops.reduce_sum(y_true) + 1e-7  # Add a small constant to avoid division by zero
#         neg_count = math_ops.reduce_sum(1 - y_true) + 1e-7  # Add a small constant to avoid division by zero

#         # Compute the number of correct predictions for positive and negative samples
#         pos_correct = math_ops.reduce_sum(y_true * y_pred)
#         neg_correct = math_ops.reduce_sum((1 - y_true) * (1 - y_pred))

#         # Compute the balanced accuracy
#         accuracy = (pos_correct / pos_count + neg_correct / neg_count) / 2

#         self.total.assign_add(accuracy)
#         self.count.assign_add(1)
#     def result(self):
#         return array_ops.identity(self.total / self.count)

#     def reset_states(self):
#         self.count.assign(0)
#         self.total.assign(0)

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
        self.model.add(Dense(64, activation='relu'))
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
    

# input data
df = pd.read_csv('data/clean_tested_molecules.csv')
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

# split data
X_train, X_test, y_ERK_train, y_ERK_test = train_test_split(X, y_ERK, test_size=0.3)
X_train2, X_test2, y_PKM_train, y_PKM_test = train_test_split(X, y_PKM, test_size=0.3)

input_size = X.shape[1]
balanced_model = BalancedClassificationModel(input_size=input_size)

weights = list(range(100, 160, 2))
baccs = []
for weight in weights:
    balanced_model.train(X_train, y_ERK_train, epochs=10, w=weight)
    bacc, cm = balanced_model.evaluate(X_test, y_ERK_test)
    baccs.append({'weight': weight, 'bacc': bacc, 'cm': cm})
# print highest bacc
best = max(baccs, key=lambda x: x['bacc'])
print(f"Best balanced accuracy: {best['bacc']} with weight {best['weight']}")
print(f"Confusion matrix: {best['cm']}")
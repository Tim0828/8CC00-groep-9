import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, classification_report
from sklearn.utils import compute_sample_weight
from imblearn.over_sampling import SMOTE
from math import sqrt

# reading in the data:
data = pd.read_csv('tested_molecules.csv')

# combine the inhibition columns into a single multi-class target
# data['inhibition_class'] = (data['PKM2_inhibition'] * 2 + data['ERK2_inhibition']).astype(int) # 0 for no inhibition, 1 for ERK2 inhibition, 
# Y_data_combined = data["inhibition_class"]

# separate features and inhibition
Y_PKM2 = data["PKM2_inhibition"]
Y_ERK2 = data["ERK2_inhibition"]
X_best_PKM2 = pd.read_csv('X_PKM2_pca.csv', header = None)
X_best_PKM2.columns = [f'PC{i+1}' for i in range(X_best_PKM2.shape[1])]
X_best_ERK2 = pd.read_csv('X_ERK2_pca.csv', header = None)
X_best_ERK2.columns = [f'PC{i+1}' for i in range(X_best_ERK2.shape[1])]

# split the data into training and testing sets
X_train_PKM2, X_test_PKM2, Y_train_PKM2, Y_test_PKM2 = train_test_split(X_best_PKM2, Y_PKM2, test_size=0.4, random_state=42)
X_train_ERK2, X_test_ERK2, Y_train_ERK2, Y_test_ERK2 = train_test_split(X_best_ERK2, Y_ERK2, test_size=0.4, random_state=42)

def Knearestneighbour(train_features, test_features, train_targets, test_targets):
        """k nearest neighbour algorithm 

        Args:
            train_features (Pandas Dataframe): dataframe containing samples and features used for testing the algorithm
            test_features (Pandas Dataframe): dataframe containing samples and features used for testing the algorithm
            train_targets (Pandas Dataframe): single column with targets used for training the algorithm
            test_targets (Pandas Dataframe): single column with targets used for testing the algorithm
        """
        # balance the training set (Hogere balanced accuracy, maar wel meer false positives)
        smote = SMOTE(random_state=42)
        x_train, y_train = smote.fit_resample(train_features, train_targets)

        #KNN classifier + cross validating until square root of N samples gives k_best = 2
        best_scores = list()
        test_neighbours = list(range(1, int(sqrt(x_train.shape[0])+1)))

        for k in test_neighbours: 

            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
            best_scores.append(scores.mean())

        k_best = test_neighbours[np.argmax(best_scores)]

        # fit KNN
        knn = KNeighborsClassifier(n_neighbors=k_best) 
        knn.fit(x_train, y_train)

        # predict on the test set
        pred_targets = knn.predict(test_features)

        # confusion matrix
        accuracy = accuracy_score(test_targets, pred_targets)
        balanced_accuracy = balanced_accuracy_score(test_targets, pred_targets)
        cm = confusion_matrix(test_targets, pred_targets)
        cmd = ConfusionMatrixDisplay(cm,)
        cmd.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced accuracy: {balanced_accuracy:.4f}")
        print(classification_report(test_targets, pred_targets))
       
Knearestneighbour(X_train_PKM2, X_test_PKM2, Y_train_PKM2, Y_test_PKM2)
Knearestneighbour(X_train_ERK2, X_test_ERK2, Y_train_ERK2, Y_test_ERK2)

print(X_best_PKM2)

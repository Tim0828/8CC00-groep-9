import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, classification_report

# reading in the data:
data = pd.read_csv('PCA_data.csv')

# combine the inhibition columns into a single multi-class target
data['inhibition_class'] = (data['PKM2_inhibition'] * 2 + data['ERK2_inhibition']).astype(int) # 0 for no inhibition, 1 for ERK2 inhibition, 

# separate features and inhibition
X_data = data.drop(columns=["PKM2_inhibition", "ERK2_inhibition", "inhibition_class"])  
Y_data_combined = data["inhibition_class"]  
Y_PKM2 = data["PKM2_inhibition"]
Y_ERK2 = data["ERK2_inhibition"]

# split the data into training and testing sets
X_train_PKM2, X_test_PKM2, Y_train_PKM2, Y_test_PKM2 = train_test_split(X_data, Y_PKM2, test_size=0.4, random_state=42)
X_train_ERK2, X_test_ERK2, Y_train_ERK2, Y_test_ERK2 = train_test_split(X_data, Y_ERK2, test_size=0.4, random_state=42)

def Knearestneighbour(train_features, test_features, train_targets, test_targets):
        """k nearest neighbour algorithm 

        Args:
            train_features (Pandas Dataframe): dataframe containing samples and features used for testing the algorithm
            test_features (Pandas Dataframe): dataframe containing samples and features used for testing the algorithm
            train_targets (Pandas Dataframe): single column with targets used for training the algorithm
            test_targets (Pandas Dataframe): single column with targets used for testing the algorithm
        """

        # train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=2) 
        knn.fit(train_features, train_targets)

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
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Balanced accuracy: {balanced_accuracy:.2f}")
        print(classification_report(test_targets, pred_targets))

Knearestneighbour(X_train_PKM2, X_test_PKM2, Y_train_PKM2, Y_test_PKM2)
Knearestneighbour(X_train_ERK2, X_test_ERK2, Y_train_ERK2, Y_test_ERK2)


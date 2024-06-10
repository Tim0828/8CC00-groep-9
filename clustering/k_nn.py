import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# reading in the data:
data = pd.read_csv('PCA_data.csv')

# combine the inhibition columns into a single multi-class target
data['inhibition_class'] = (data['PKM2_inhibition'] * 2 + data['ERK2_inhibition']).astype(int) # 0 for no inhibition, 1 for ERK2 inhibition, 

# separate features and inhibition
X_data = data.drop(columns=["PKM2_inhibition", "ERK2_inhibition", "inhibition_class"])  
Y_data = data["inhibition_class"]  


# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.4, random_state=42)

def Knearestneighbour(train_features, test_features, train_targets, test_targets):
        """k nearest neighbour algorithm 

        Args:
            train_features (Pandas Dataframe): dataframe containing samples and features used for testing the algorithm
            test_features (Pandas Dataframe): dataframe containing samples and features used for testing the algorithm
            train_targets (Pandas Dataframe): single column with targets used for training the algorithm
            test_targets (Pandas Dataframe): single column with targets used for testing the algorithm
        """

        # train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=4) 
        knn.fit(train_features, train_targets)

        # predict on the test set
        pred_targets = knn.predict(test_features)

        # confusion matrix
        accuracy = accuracy_score(test_targets, pred_targets)
        cm = confusion_matrix(test_targets, pred_targets)
        cmd = ConfusionMatrixDisplay(cm, display_labels=[0, 1, 2, 3])
        cmd.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()
        print(f"Accuracy: {accuracy:.2f}")

Knearestneighbour(X_train, X_test, y_train, y_test)


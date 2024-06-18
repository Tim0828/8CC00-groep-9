import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, balanced_accuracy_score
import pandas as pd
import csv

class Reg:
    def __init__(self, X_file, y_file):
        self.X = pd.read_csv(X_file)
        self.y = pd.read_csv(y_file)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        self.highest_BAcc_result = float('-inf')
        self.y_train = self.y_train.values.ravel()
        self.y_test = self.y_test.values.ravel()

    def logistic_regression(self, min_weight, max_weight, step_weight):
        for i in range(min_weight, max_weight, step_weight):

            logr_model=LogisticRegression(class_weight={0: 1, 1: i}) #weights added to make sure 1 has a higher chance
            logr_model.fit(self.X_train,self.y_train)

            y_pred = logr_model.predict(self.X_test)
            y_prob = logr_model.predict_proba(self.X_test)[:, 1]
            balanced_accuracy = balanced_accuracy_score(self.y_test, y_pred)

            if balanced_accuracy > self.highest_BAcc_result:
                self.highest_BAcc_result = balanced_accuracy
                self.best_y_pred = y_pred
                self.best_y_prob = y_prob
                self.best_model = logr_model
                self.best_weight = i
        print("Results of logistic regression for prediction of ERK_2 inhibition") 
        print("Balanced accuracy: {:.2f}%".format(self.highest_BAcc_result  * 100))
        print("Weight: ", self.best_weight)
        print("Classification Report:\n", classification_report(self.y_test, self.best_y_pred))
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, self.best_y_pred))

    def plot_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.y_test, self.best_y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
    
    def predict(self, X):
        return self.best_model.predict(X)

def PCA_untested_data(df):
    from sklearn.decomposition import PCA
    # minmax
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    pca = PCA(n_components=20)
    principalComponents = pca.fit_transform(df)
    return principalComponents

def load_untested_data():
    df = pd.read_csv("untested_molecules_with_descriptors.csv")
    results = df[['SMILES', 'PKM2_inhibition', 'ERK2_inhibition']]
    df = df.drop(columns=['SMILES', 'PKM2_inhibition', 'ERK2_inhibition'])
    df = PCA_untested_data(df)
    return df, results

def main():
    "Train model and predict both inhibitors"
    reg_ERK = Reg("X_ERK2_pca.csv", "y_ERK2.csv")
    reg_PKM = Reg("X_PKM2_pca.csv", "y_PKM2.csv")
    reg_ERK.logistic_regression(1, 150, 1)
    reg_PKM.logistic_regression(1, 150, 1)
    # reg_ERK.plot_roc_curve()
    # reg_PKM.plot_roc_curve()
    X_untest, _ = load_untested_data()
    y_ERK = reg_ERK.predict(X_untest)
    y_PKM = reg_PKM.predict(X_untest)
    # conversion to integers
    y_ERK = y_ERK.astype(int)
    y_PKM = y_PKM.astype(int)

    # WHY DID THE FORMATTING HAD TO BE LIKE THIS
    # conversion to string
    y_ERK = y_ERK.astype(str)
    y_PKM = y_PKM.astype(str)
    # save results
    
    df = pd.read_csv('untested_molecules_format.csv')
    df['PKM2_inhibition'] = y_PKM
    df['ERK2_inhibition'] = y_ERK
    df.to_csv('untested_molecules.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == "__main__":
    main()
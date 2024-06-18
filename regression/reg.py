import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, balanced_accuracy_score
import pandas as pd

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

def main():
    reg = Reg("X_PKM2_pca.csv", "y_PKM2.csv")
    reg.logistic_regression(50, 150, 2)
    reg.plot_roc_curve()

if __name__ == "__main__":
    main()

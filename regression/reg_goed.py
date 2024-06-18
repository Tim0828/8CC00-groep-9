import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import pandas as pd
import seaborn as sns

class Reg:
    def __init__(self, X_file, y_file, label):
        self.X = pd.read_csv(X_file)
        self.y = pd.read_csv(y_file)
        self.label = label
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
        print(f"Results of logistic regression for prediction of {self.label} inhibition")
        print("Balanced accuracy: {:.2f}%".format(self.highest_BAcc_result  * 100))
        print("Weight: ", self.best_weight)
        print("Classification Report:\n", classification_report(self.y_test, self.best_y_pred))
        self.plot_confusion_matrix()
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, self.best_y_pred))


    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.best_y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=['Non-inhibitor', 'Inhibitor'], 
                    yticklabels=['Non-inhibitor', 'Inhibitor'],
                    annot_kws={"size": 16})
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix for {self.label}')
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
    reg_ERK = Reg("X_ERK2_pca.csv", "y_ERK2.csv", "ERK_2")
    reg_PKM = Reg("X_PKM2_pca.csv", "y_PKM2.csv", "PKM_2")
    reg_ERK.logistic_regression(1, 150, 1)
    reg_PKM.logistic_regression(1, 150, 1)
    X_untest, result_table = load_untested_data()
    y_ERK = reg_ERK.predict(X_untest)
    y_PKM = reg_PKM.predict(X_untest)
    # conversion to integers
    y_ERK = y_ERK.astype(int)
    y_PKM = y_PKM.astype(int)
    result_table['ERK2_inhibition'] = y_ERK
    result_table['PKM2_inhibition'] = y_PKM
    result_table.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()

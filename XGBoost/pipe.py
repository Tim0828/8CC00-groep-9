import xgboost as xgb
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

class Pipeline:
    def __init__(self, mod='classifier'):
        if mod == 'classifier':
            self.model = xgb.XGBClassifier(objective='binary:logistic')
        else:
            self.model = xgb.XGBRegressor(objective='binary:logistic')

    def load_data(self, X_file):
        if 'ERK' in X_file:
            X = pd.read_csv(X_file)
            y = pd.read_csv('y_ERK2.csv')
            return X, y
        else:
            X = pd.read_csv(X_file)
            y = pd.read_csv('y_PKM2.csv')
            return X, y
        
    def test_train(self, X_filename):
        """
        split the data into training and testing sets keeping in mind the class distribution
        """

        # load the data
        X, y = self.load_data(X_filename)
        # split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y) # stratify to keep the class distribution same in both training and testing sets

        return X_train, X_test, y_train, y_test

    def train(self, X, y):
        self.model.fit(X, y)
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """Evaluate model by balanced accuracy score"""
        y_pred = self.predict(X)
        return balanced_accuracy_score(y, y_pred)
    
    def conf_matrix(self, X, y):
        """Compute and print confusion matrix"""
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred)
        print(cm)

# define pipeline
pipeline = Pipeline("classifier")
# load data
X_file = 'X_PKM2_pca.csv'
X_train, X_test, y_train, y_test = pipeline.test_train(X_file)

# train model
model = pipeline.train(X_train, y_train)

# evaluate model
score = pipeline.evaluate(X_test, y_test)
print(f'Balanced accuracy: {score:.2f}')
pipeline.conf_matrix(X_test, y_test)
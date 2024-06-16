import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

def get_data(infile):
    """Retrieves the PCA data and splits into training and testing data.

    Args:
        infile (str): Location/name of the PCA data file as string.

    Returns:
        X_train, X_test, Y_train, Y_test: Data split up into training and test data.
    """
    # reading the data: 
    df_pca = pd.read_csv(infile)

    # splitting the data into test and training data:
    df_X = df_pca.drop(["PKM2_inhibition","ERK2_inhibition"],axis=1)
    df_Y = df_pca[["PKM2_inhibition","ERK2_inhibition"]]
    X_train, X_test, Y_train, Y_test = train_test_split(df_X,df_Y,test_size=0.3)
    
    return X_train, X_test, Y_train, Y_test

def train_random_forest(X_train, Y_train):
    """trains a random forest model based on the training data using a grid search to find 
    the best hyperparameters.

    Args:
        X_train (Pandas DataFrame): PCA training data
        Y_train (Pandas DataFrame): Inhibition training data.

    Returns:
        best_model: Best random forest classifier model based on training data and grid search.
    """
    # applying SMOTE:
    smote = SMOTE()
    over_X_train, over_Y_train = smote.fit_resample(X= X_train, y=Y_train)
    
    # Random Forest:
    classifier = RandomForestClassifier()
    
    # using grid search to find best hyperparameters for the random forest:
    param_grid = {
        'n_estimators': list(range(100,2000,100)),
        'max_depth': [1,5,10,20,30,40,50,60,70,80,90,100,None],
        'max_features': ['auto','sqrt'],
        'min_samples_split': [2,5,10],
        'min_samples_leaf': [1,2,4,10]
        }
    grid_search = RandomizedSearchCV(estimator=classifier, param_distributions=param_grid, scoring='f1', n_jobs=-1, cv=None, n_iter=50)
    grid_search.fit(over_X_train, over_Y_train)

    best_model = grid_search.best_estimator_
    return best_model

def evaluate_random_forest(rf_model, X_test, Y_test):
    """Evaluates the random forest model based on the test data.

    Args:
        rf_model (sklearn random forest classifier): The to be tested random forest classifier.
        X_test (Pandas DataFrame): PCA test data.
        Y_test (Array): Inhibition test data.
    """
    # prediction based on model:
    Y_predict = rf_model.predict(X_test)
    
    # plotting confusion matrix:
    cm = confusion_matrix(Y_test,Y_predict)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title("Standard Random Forest + SMOTE")
    plt.show(block = True)
    
    # Classfication report and balanced accuracy
    print("Classification report:\n", classification_report(Y_test, Y_predict))
    print("Balanced Accuracy: ", balanced_accuracy_score(Y_test,Y_predict))
    print("Total Accuracy: ", accuracy_score(Y_test,Y_predict))
    
X_train, X_test, Y_train, Y_test = get_data(r"C:\Users\20223319\OneDrive - TU Eindhoven\Bestanden\Jaar 2\Q4\Advanced Programming\Group Assignment\data\PCA_data.csv")

Y_train_PKM2 = Y_train["PKM2_inhibition"]
Y_train_ERK2 = Y_train["ERK2_inhibition"]
Y_test_PKM2 = Y_test["PKM2_inhibition"]
Y_test_ERK2 = Y_test["ERK2_inhibition"]

rf_model_PKM2 = train_random_forest(X_train,Y_train_PKM2)
rf_model_ERK2 = train_random_forest(X_train,Y_train_ERK2)

evaluate_random_forest(rf_model_PKM2, X_test,Y_test_PKM2)
evaluate_random_forest(rf_model_ERK2, X_test,Y_test_ERK2)
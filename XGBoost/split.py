from sklearn.model_selection import train_test_split
import pandas as pd
def load_data(X_file):
        if 'ERK' in X_file:
            X = pd.read_csv(X_file)
            y = pd.read_csv('y_ERK2.csv')
            return X, y
        else:
            X = pd.read_csv(X_file)
            y = pd.read_csv('y_PKM2.csv')
            return X, y

def test_train(X_filename):
    """
    split the data into training and testing sets keeping in mind the class distribution
    """

    # load the data
    X, y = load_data(X_filename)
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y) # stratify to keep the class distribution same in both training and testing sets

    # save the data
    X_train.to_csv(f"train_{X_filename}", index=False)
    X_test.to_csv(f"test_{X_filename}", index=False)
    y_train.to_csv(f"ytrain_{X_filename}", index=False)
    y_test.to_csv(f"ytest_{X_filename}", index=False)
    return

# test the function'
test_train('X_ERK2_pca.csv')
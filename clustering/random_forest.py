import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE

# reading the data: 
df_pca = pd.read_csv(r"C:\Users\20223319\OneDrive - TU Eindhoven\Bestanden\Jaar 2\Q4\Advanced Programming\Group Assignment\data\PCA_data.csv")

# splitting the data into test and training data:
df_X = df_pca.drop(["PKM2_inhibition","ERK2_inhibition"],axis=1)
df_Y = df_pca[["PKM2_inhibition","ERK2_inhibition"]]
X_train, X_test, Y_train, Y_test = train_test_split(df_X,df_Y,test_size=0.3)

##### Standard Random Forest #####

# fitting data:
classifier = RandomForestClassifier(100)
rf = classifier.fit(X_train,Y_train)

# evaluating the result:
Y_predict = rf.predict(X_test)
accuracy = accuracy_score(Y_test,Y_predict)

cm = confusion_matrix(Y_test["PKM2_inhibition"], Y_predict[:,0])
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Standard Random Forest (PKM2)")
plt.show(block = False)

cm = confusion_matrix(Y_test["ERK2_inhibition"], Y_predict[:,0])
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Standard Random Forest (ERK2)")
plt.show(block = False)


##### Balanced Random Forest #####

# fitting data:
balanced_classifier = BalancedRandomForestClassifier(100)
balanced_rf = balanced_classifier.fit(X_train,Y_train["PKM2_inhibition"])

# evaluating the result:
Y_predict = balanced_rf.predict(X_test)
cm = confusion_matrix(Y_test["PKM2_inhibition"], Y_predict)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Balanced Random Forest (PKM2)")
plt.show(block = False)

# fitting data:
balanced_rf = balanced_classifier.fit(X_train,Y_train["ERK2_inhibition"])

# evaluating the result:
Y_predict = balanced_rf.predict(X_test)
cm = confusion_matrix(Y_test["ERK2_inhibition"], Y_predict)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Balanced Random Forest (ERK2)")
plt.show(block = False)


##### Standard Random Forest + SMOTE #####

# SMOTE (PKM2):
over_sample = SMOTE()
over_X, over_Y = over_sample.fit_resample(X=df_X, y=df_Y["PKM2_inhibition"])
over_X_train, over_X_test, over_Y_train, over_Y_test = train_test_split(over_X,over_Y,test_size=0.3)

# fitting data (PKM2):
classifier = RandomForestClassifier(100)
rf = classifier.fit(over_X_train,over_Y_train)

# evaluating the result (PKM2):
Y_predict = rf.predict(over_X_test)

cm = confusion_matrix(over_Y_test,Y_predict)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Standard Random Forest + SMOTE (PKM2)")
plt.show(block = False)

# SMOTE (ERK2):
over_sample = SMOTE()
over_X, over_Y = over_sample.fit_resample(X=df_X, y=df_Y["ERK2_inhibition"])
over_X_train, over_X_test, over_Y_train, over_Y_test = train_test_split(over_X,over_Y,test_size=0.3)

# fitting data (ERK2):
classifier = RandomForestClassifier(100)
rf = classifier.fit(over_X_train,over_Y_train)

# evaluating the result (EKR2):
Y_predict = rf.predict(over_X_test)

cm = confusion_matrix(over_Y_test,Y_predict)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Standard Random Forest + SMOTE (ERK2)")
plt.show(block = True)




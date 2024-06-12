import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

# Load data
file_path = "C:/Users/20201954/OneDrive - TU Eindhoven/2023/2023-4/8CC00 Advanced programming/tested_molecules_with_descriptors.csv"
data = pd.read_csv(file_path)

# Separate descriptors and targets
descriptors = data.columns[3:]  # Excluding SMILES and inhibition columns

# Example for PKM2
X = data[descriptors]
y_PKM2 = data['PKM2_inhibition']
y_ERK2 = data['ERK2_inhibition']

# Remove constant features
constant_filter = VarianceThreshold(threshold=0.0)
X_constant_removed = constant_filter.fit_transform(X)
columns_kept = X.columns[constant_filter.get_support()]
X_filtered = pd.DataFrame(X_constant_removed, columns=columns_kept)

# Calculate correlation matrix
corr_matrix = X_filtered.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find index of feature columns with correlation greater than 0.95
threshold_corr = 0.95
to_drop = [column for column in upper.columns if any(upper[column] > threshold_corr)]

# Drop highly correlated features
X_filtered = X_filtered.drop(columns=to_drop)

# Univariate feature selection for PKM2
selector_PKM2 = SelectKBest(score_func=f_classif, k=50)
X_best_PKM2 = selector_PKM2.fit_transform(X_filtered, y_PKM2)
selected_features_PKM2 = X_filtered.columns[selector_PKM2.get_support()]

# Univariate feature selection for ERK2
selector_ERK2 = SelectKBest(score_func=f_classif, k=50)
X_best_ERK2 = selector_ERK2.fit_transform(X_filtered, y_ERK2)
selected_features_ERK2 = X_filtered.columns[selector_ERK2.get_support()]

# Split into training and testing sets for PKM2
X_train_PKM2, X_test_PKM2, y_train_PKM2, y_test_PKM2 = train_test_split(X_best_PKM2, y_PKM2, test_size=0.2, random_state=42)
# Split into training and testing sets for ERK2
X_train_ERK2, X_test_ERK2, y_train_ERK2, y_test_ERK2 = train_test_split(X_best_ERK2, y_ERK2, test_size=0.2, random_state=42)

# Model training with Random Forest for PKM2
model_PKM2 = RandomForestClassifier(random_state=42)
model_PKM2.fit(X_train_PKM2, y_train_PKM2)
# Model training with Random Forest for ERK2
model_ERK2 = RandomForestClassifier(random_state=42)
model_ERK2.fit(X_train_ERK2, y_train_ERK2)

# Feature importance analysis for PKM2
feature_importances_PKM2 = model_PKM2.feature_importances_
importance_df_PKM2 = pd.DataFrame({'Descriptor': selected_features_PKM2, 'Importance': feature_importances_PKM2})
importance_df_PKM2 = importance_df_PKM2.sort_values(by='Importance', ascending=False)

# Plot feature importances for PKM2
plt.figure(figsize=(10, 6))
plt.bar(importance_df_PKM2['Descriptor'].head(10), importance_df_PKM2['Importance'].head(10))
plt.xticks(rotation=90)
plt.title('Top 10 Feature Importances for PKM2 Inhibition')
plt.xlabel('Descriptor')
plt.ylabel('Importance')
plt.show()

# Determine a good threshold for PKM2
cumulative_importance_PKM2 = importance_df_PKM2['Importance'].cumsum()
plt.figure(figsize=(10, 6))
plt.plot(cumulative_importance_PKM2)
plt.axhline(y=0.95, color='r', linestyle='--')
plt.title('Cumulative Feature Importance for PKM2 Inhibition')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Importance')
plt.show()

# Choose a threshold for PKM2
threshold_PKM2 = 0.01  # This value can be adjusted based on the plot
important_features_df_PKM2 = importance_df_PKM2[importance_df_PKM2['Importance'] >= threshold_PKM2]

# Print selected important features with their importance values for PKM2
print(f"Selected important features with threshold {threshold_PKM2} for PKM2:")
print(important_features_df_PKM2)

# Evaluate the model for PKM2
y_pred_PKM2 = model_PKM2.predict(X_test_PKM2)
accuracy_PKM2 = accuracy_score(y_test_PKM2, y_pred_PKM2)
conf_matrix_PKM2 = confusion_matrix(y_test_PKM2, y_pred_PKM2)
class_report_PKM2 = classification_report(y_test_PKM2, y_pred_PKM2, zero_division=0)

print(f"Accuracy for PKM2: {accuracy_PKM2}")
print("Confusion Matrix for PKM2:")
print(conf_matrix_PKM2)
print("Classification Report for PKM2:")
print(class_report_PKM2)

# Repeat the same process for ERK2
# Feature importance analysis for ERK2
feature_importances_ERK2 = model_ERK2.feature_importances_
importance_df_ERK2 = pd.DataFrame({'Descriptor': selected_features_ERK2, 'Importance': feature_importances_ERK2})
importance_df_ERK2 = importance_df_ERK2.sort_values(by='Importance', ascending=False)

# Plot feature importances for ERK2
plt.figure(figsize=(10, 6))
plt.bar(importance_df_ERK2['Descriptor'].head(10), importance_df_ERK2['Importance'].head(10))
plt.xticks(rotation=90)
plt.title('Top 10 Feature Importances for ERK2 Inhibition')
plt.xlabel('Descriptor')
plt.ylabel('Importance')
plt.show()

# Determine a good threshold for ERK2
cumulative_importance_ERK2 = importance_df_ERK2['Importance'].cumsum()
plt.figure(figsize=(10, 6))
plt.plot(cumulative_importance_ERK2)
plt.axhline(y=0.95, color='r', linestyle='--')
plt.title('Cumulative Feature Importance for ERK2 Inhibition')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Importance')
plt.show()

# Choose a threshold for ERK2
threshold_ERK2 = 0.01  # This value can be adjusted based on the plot
important_features_df_ERK2 = importance_df_ERK2[importance_df_ERK2['Importance'] >= threshold_ERK2]

# Print selected important features with their importance values for ERK2
print(f"Selected important features with threshold {threshold_ERK2} for ERK2:")
print(important_features_df_ERK2)

# Evaluate the model for ERK2
y_pred_ERK2 = model_ERK2.predict(X_test_ERK2)
accuracy_ERK2 = accuracy_score(y_test_ERK2, y_pred_ERK2)
conf_matrix_ERK2 = confusion_matrix(y_test_ERK2, y_pred_ERK2)
class_report_ERK2 = classification_report(y_test_ERK2, y_pred_ERK2, zero_division)
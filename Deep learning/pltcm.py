# plot confusion matrix
cm_PKM = [[255, 19], [1, 4]]
cm_ERK = [[116, 154], [0, 9]]

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm_PKM, annot=True, fmt='d', cmap='Blues', xticklabels=['Inhibitor', 'non-Inhibitor'], yticklabels=['Inhibitor', 'Non-Inhibitor'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('PKM2 Confusion Matrix')
plt.show()


sns.heatmap(cm_ERK, annot=True, fmt='d', cmap='Blues', xticklabels=['Inhibitor', 'non-Inhibitor'], yticklabels=['Inhibitor', 'Non-Inhibitor'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('ERK2 Confusion Matrix')
plt.show()
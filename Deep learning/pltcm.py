# plot confusion matrix
cm_PKM = [[255, 19], [1, 4]]
cm_ERK = [[116, 154], [0, 9]]

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm_PKM, annot=True, fmt='d', cmap='Blues', xticklabels=['True', 'False'], yticklabels=['True', 'False'])
plt.xlabel('Predicted')
plt.ylabel('Ground Truth')
plt.title('PKM2 Confusion Matrix')
plt.show()


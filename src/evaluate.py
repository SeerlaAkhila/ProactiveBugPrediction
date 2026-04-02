# src/evaluate.py

import matplotlib.pyplot as plt

# Your results
models = ['Baseline', 'Improved']
recall = [0.57, 0.66]
accuracy = [0.83, 0.80]

# Recall Graph
plt.bar(models, recall)
plt.title("Recall Comparison (Bug Detection)")
plt.ylabel("Recall")
plt.show()

# Accuracy Graph
plt.bar(models, accuracy)
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()
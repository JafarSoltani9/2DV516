import numpy as np
import pandas as pd
from MachineLearningModel import LogisticRegression
from ForwardSelection import ForwardSelection
from ROCAnalysis import ROCAnalysis

# Let's assume the CSV has been loaded into a pandas DataFrame: df
df = pd.read_csv('datasets/heart_disease_cleveland.csv')

# Normalizing the data (Feature Scaling)
X = df.iloc[:, :-1].values  # all rows, all columns except the last
y = df.iloc[:, -1].values   # all rows, last column

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_normalized = (X - X_mean) / X_std

# Splitting the data into 80% training and 20% validation
np.random.seed(42)  # for reproducibility
indices = np.arange(X_normalized.shape[0])
np.random.shuffle(indices)

split_index = int(0.8 * len(X_normalized))
X_train, X_val = X_normalized[indices[:split_index]], X_normalized[indices[split_index:]]
y_train, y_val = y[indices[:split_index]], y[indices[split_index:]]


# Initialize your ForwardSelection with your logistic regression model
logistic_model = LogisticRegression()  # Your own logistic regression implementation
forward_selector = ForwardSelection(X_train, y_train, logistic_model)

# Perform forward feature selection
forward_selector.forward_selection()

# Fit the model using the selected features
forward_selector.fit()

# Make predictions on the validation set
y_val_pred = forward_selector.predict(X_val[:, forward_selector.selected_features])

# Evaluate the model's performance using the ROCAnalysis class
roc_analysis = ROCAnalysis(y_val_pred, y_val)
print("TP Rate:", roc_analysis.tp_rate())
print("FP Rate:", roc_analysis.fp_rate())
print("Precision:", roc_analysis.precision())
print("F-Score:", roc_analysis.f_score())

# Report selected features and their impact
print("Selected features:", forward_selector.selected_features)

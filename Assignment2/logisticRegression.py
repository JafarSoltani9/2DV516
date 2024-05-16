import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from MachineLearningModel import LogisticRegression, NonLinearLogisticRegression

# Step 1: Read and normalize the data
data = pd.read_csv('datasets/banknote_authentication.csv', header=None)
X = data.iloc[:, :2]
y = data.iloc[ :, 2]

X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
plt.figure(figsize=(8, 6))
plt.scatter(X_normalized[y == 0].iloc[:, 0], X_normalized[y == 0].iloc[:, 1], color='blue', label='FAKE', alpha=0.5)
plt.scatter(X_normalized[y == 1].iloc[:, 0], X_normalized[y == 1].iloc[:, 1], color='red', label='NOT FAKE', alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Banknote Authentication')


# Step 2: Separate a validation set
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2)


# Step 3: Find appropriate learning rate and number of iterations
learning_rate = 0.01
num_iterations = 1000

# Step 4: Method to split dataset into training and test sets
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
# Step 5: Repeat experiments 20 times
num_experiments = 20
accuracies_linear = []
accuracies_nonlinear = []

for _ in range(num_experiments):
    X_train, X_test, y_train, y_test = split_data(X_normalized, y)
    
    # Linear Logistic Regression
    model_linear = LogisticRegression(learning_rate=learning_rate, num_iterations=num_iterations)
    model_linear.fit(X_train, y_train)
    y_pred_linear = np.round(model_linear.predict(X_test))
    accuracy_linear = accuracy_score(y_test, y_pred_linear)
    accuracies_linear.append(accuracy_linear)
    
    # Nonlinear Logistic Regression
    model_nonlinear = NonLinearLogisticRegression(learning_rate=learning_rate, num_iterations=num_iterations)
    X_train_poly = model_nonlinear._polynomial_features(X_train, degree=2)
    X_test_poly = model_nonlinear._polynomial_features(X_test, degree=2)
    model_nonlinear.fit(X_train_poly, y_train)
    y_pred_nonlinear = np.round(model_nonlinear.predict(X_test_poly))
    accuracy_nonlinear = accuracy_score(y_test, y_pred_nonlinear)
    accuracies_nonlinear.append(accuracy_nonlinear)










plt.legend()
plt.show()
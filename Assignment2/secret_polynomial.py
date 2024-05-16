import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from MachineLearningModel import RegressionModelNormalEquation

# Step 1: Read the dataset and split into training and test sets
data = pd.read_csv('datasets/secret_polynomial.csv')
X = data['X'].values.reshape(-1, 1)
y = data['y'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Plotting the dataset, training set, and test set
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].scatter(X, y, alpha=0.5)
axs[0].set_title('Dataset')
axs[0].set_xlabel('X')
axs[0].set_ylabel('y')

axs[1].scatter(X_train, y_train, alpha=0.5)
axs[1].set_title('Training Set')
axs[1].set_xlabel('X')
axs[1].set_ylabel('y')

axs[2].scatter(X_test, y_test, alpha=0.5)
axs[2].set_title('Test Set')
axs[2].set_xlabel('X')
axs[2].set_ylabel('y')

# Step 2: Fit and plot polynomial models for degrees d ∈ [1, 6]
degrees = range(1, 7)
train_errors = []
test_errors = []

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
for i, degree in enumerate(degrees):
    # Fit model
    model = RegressionModelNormalEquation(degree=degree)
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate errors
    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)
    train_errors.append(train_error)
    test_errors.append(test_error)
    
    # Plotting
    axs[i//3, i%3].scatter(X_train, y_train, alpha=0.5, label='Training data')
    axs[i//3, i%3].scatter(X_test, y_test, alpha=0.5, label='Test data')
    axs[i//3, i%3].plot(sorted(X_train), model.predict(sorted(X_train)), color='red', label='Prediction')
    axs[i//3, i%3].set_title(f'Degree {degree}')
    axs[i//3, i%3].set_xlabel('X')
    axs[i//3, i%3].set_ylabel('y')
    axs[i//3, i%3].legend()


# Step 3: Repeated runs with shuffled data
num_runs = 20
train_errors_avg = []
test_errors_avg = []

for _ in range(num_runs):
    # Shuffle data
    shuffled_indices = np.random.permutation(len(X))
    X_shuffled = X[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    
    # Split into training and test sets
    X_train_shuffled, X_test_shuffled, y_train_shuffled, y_test_shuffled = train_test_split(
        X_shuffled, y_shuffled, test_size=0.2, random_state=42)
    
    # Fit and evaluate model
    train_errors_run = []
    test_errors_run = []
    for degree in degrees:
        model = RegressionModelNormalEquation(degree=degree)
        model.fit(X_train_shuffled, y_train_shuffled)
        
        y_train_pred = model.predict(X_train_shuffled)
        y_test_pred = model.predict(X_test_shuffled)
        
        train_error = mean_squared_error(y_train_shuffled, y_train_pred)
        test_error = mean_squared_error(y_test_shuffled, y_test_pred)
        
        train_errors_run.append(train_error)
        test_errors_run.append(test_error)
    
    train_errors_avg.append(train_errors_run)
    test_errors_avg.append(test_errors_run)

train_errors_avg = np.mean(train_errors_avg, axis=0)
test_errors_avg = np.mean(test_errors_avg, axis=0)

print("Average Train Errors:", train_errors_avg)
print("Average Test Errors:", test_errors_avg)
# After computing the average errors for each degree:
# After computing the average errors for each degree:

# Find the index of the degree with the minimum average test error
best_degree_index = np.argmin(test_errors_avg)
best_degree = degrees[best_degree_index]
best_avg_test_error = test_errors_avg[best_degree_index]

print(f"The best polynomial degree based on the average test errors is: {best_degree}")
print('The model with degree 4 has the lowest average test error (4869.68) among all the degrees considered.' +
    'Even though the train error might still decrease for degrees higher than 4, '+
    'the test error starts to increase after degree 4. This indicates that models with degrees higher than 4 are starting ' +
    'to overfit the training data: they are capturing noise or patterns that do not generalize to the test data.')

print(f'Best after shuffled: (degree 4)  {best_avg_test_error}')

plt.tight_layout()
plt.show()
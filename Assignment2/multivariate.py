import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
from MachineLearningModel import RegressionModelGradientDescent, RegressionModelNormalEquation

# Step 1: Read the dataset and store values
data = pd.read_csv('datasets/housing-boston.csv')
X = data[['INDUS', 'RM']].values
y = data['PRICE'].values

# Step 2: Plot the dataset
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

for i in range(2):
    axs[i].scatter(X[:, i], y, alpha=0.5)
    axs[i].set_xlabel(['INDUS', 'RM'][i])
    axs[i].set_ylabel('PRICE')


# Step 3: Use RegressionModelNormalEquation
model = RegressionModelNormalEquation(degree=1)
model.fit(X, y)
print("Values for beta:", model.beta)
print("Cost:", model.evaluate(X, y))
predicted_value = model.predict(np.array([[2.31, 6.575]]))
print("Predicted value:", predicted_value)

# Step 4: Normalize input features and use RegressionModelNormalEquation again
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
model.fit(X_normalized, y)
print("\nValues for beta after normalization:", model.beta)
print("Cost after normalization:", model.evaluate(X_normalized, y))


# Step 5: Compare cost function evolution with and without normalization using Gradient Descent
model_gradient_descent = RegressionModelGradientDescent(degree=1, learning_rate=0.001, num_iterations=3000)

# Without normalization
model_gradient_descent.fit(X, y)
cost_history_non_normalized = model_gradient_descent.cost_history

model_gradient_descent_non = RegressionModelGradientDescent(degree=1, learning_rate=0.001, num_iterations=3000)

# With normalization
model_gradient_descent_non.fit(X_normalized, y)
cost_history_normalized = model_gradient_descent_non.cost_history

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(cost_history_non_normalized)
axs[0].set_title('Non-normalized')
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel('Cost')

axs[1].plot(cost_history_normalized)
axs[1].set_title('Normalized')
axs[1].set_xlabel('Iterations')
axs[1].set_ylabel('Cost')



# Step 6: Find learning rate and number of iterations
learning_rates = [0.001, 0.001, 0.01,0.1, 0.02]
num_iterations = [1000, 2000, 3000, 4000, 5000]
lrs = []

print("\n\nStarting loop with learning rates and iterations...")
for lr in learning_rates:
    for it in num_iterations:
        print(f"Testing with Learning Rate: {lr}, Iterations: {it}")
        model_gradient_descent = RegressionModelGradientDescent(degree=1, learning_rate=lr, num_iterations=it)
        model_gradient_descent.fit(X_normalized, y)
        final_cost = model_gradient_descent.cost_history[-1]
        eval_cost = model.evaluate(X_normalized, y) # Evaluate once to use in comparison and print
        d = model_gradient_descent.evaluate(X_normalized, y)
 
        if abs((d - eval_cost) /eval_cost ) <= 0.01:
            lrs.append( (lr, it) )
            print("Learning Rate:", lr)
            print("Number of Iterations:", it)
            break
        else:
            print(f"Not met for LR {lr} and Iterations {it})")
# Prepare the plot
plt.figure(figsize=(10, 6))

for lr in learning_rates:
    for it in num_iterations:
        plt.scatter(lr, it, color='grey', marker='o', s=100) 
for lr, it in lrs:
    plt.scatter(lr, it, color='green', marker='o', s=100)


plt.title('Learning Rate and Iteration')
plt.xlabel('Learning Rate')
plt.ylabel('Number of Iterations') 
plt.grid(True)  

print(lrs)
print('\n\nDescribe what is happening and why this happens (i.e., using or not normalization)?\nNormalization of the input usually leads to faster and more stable convergence for optimization algorithms.' +
    ' Right graph with normalized data shows a smoother and faster decrease in cost compared to left graph without normalization.' +
    'This is because normalization ensures that all features contribute to the optimization process, preventing large ranges of variation of the features '+ 
    'values from causing irregular gradient updates and numerical instability.')
plt.tight_layout()
plt.show()
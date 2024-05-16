from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class MachineLearningModel(ABC):
    """
    Abstract base class for machine learning models.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        pass
    def _polynomial_features(self, X, degree):
        """
            Generate polynomial features from the input features.
            Check the slides for hints on how to implement this one. 
            This method is used by the regression models and must work
            for any degree polynomial
            Parameters:
            X (array-like): Features of the data.

            Returns:
            X_poly (array-like): Polynomial features.
        """
        X_poly = np.ones((len(X), 1))
        for d in range(1, degree + 1):
            X_poly = np.concatenate((X_poly, np.power(X, d)), axis=1)
        return X_poly

class RegressionModelNormalEquation(MachineLearningModel):
    """
    Class for regression models using the Normal Equation for polynomial regression.
    """

    def __init__(self, degree):
        """
        Initialize the model with the specified polynomial degree.

        Parameters:
        degree (int): Degree of the polynomial features.
        """
        self.degree = degree
        self.beta = None

    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        X_poly = self._polynomial_features(X, self.degree)
        self.beta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        X_poly = self._polynomial_features(X, self.degree)
        return X_poly @ self.beta

    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (MSE).
        """
        predictions = self.predict(X)
        mse = np.mean((y - predictions)**2)
        return mse

class RegressionModelGradientDescent(MachineLearningModel):
    """
    Class for regression models using gradient descent optimization.
    """

    def __init__(self, degree, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the model with the specified parameters.

        Parameters:
        degree (int): Degree of the polynomial features.
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations for gradient descent.
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.beta = None
        self.cost_history = []

    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        X_poly = self._polynomial_features(X, self.degree)
        m, n = X_poly.shape
        self.beta = np.zeros(n)
        
        for _ in range(self.num_iterations):
            predictions = X_poly @ self.beta
            error = predictions - y
            gradient = (1/m) * (X_poly.T @ error)
            self.beta -= self.learning_rate * gradient
            cost = np.sum((error ** 2)) / (2 * m)
            self.cost_history.append(cost)

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        X_poly = self._polynomial_features(X, self.degree)
        return X_poly @ self.beta

    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (MSE).
        """
        predictions = self.predict(X)
        mse = np.mean((y - predictions)**2)
        return mse
    

class LogisticRegression:
    """
    Logistic Regression model using gradient descent optimization.
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the logistic regression model.

        Parameters:
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.beta = None
        self.cost_history = []

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        m, n = X.shape
        self.beta = np.zeros(n)
        
        for _ in range(self.num_iterations):
            z = np.dot(X, self.beta)
            h = self._sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / m
            self.beta -= self.learning_rate * gradient
            
            # Calculate and store cost
            cost = self._cost_function(X, y, h)
            self.cost_history.append(cost)

    def predict(self, X):
        """
        Make predictions using the trained logistic regression model.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted probabilities.
        """
        z = np.dot(X, self.beta)
        return self._sigmoid(z)

    def evaluate(self, X, y):
        """
        Evaluate the logistic regression model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (e.g., accuracy).
        """
        predictions = self.predict(X)
        return (-1 / len(y)) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))


    def _sigmoid(self, z):
        """
        Sigmoid function.

        Parameters:
        z (array-like): Input to the sigmoid function.

        Returns:
        result (array-like): Output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    def _cost_function(self, X, y, h):
        """
        Compute the logistic regression cost function.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        m = len(y)
        cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost
    
class NonLinearLogisticRegression:
    """
    Nonlinear Logistic Regression model using gradient descent optimization.
    It works for 2 features (when creating the variable interactions)
    """

    def __init__(self, degree=2, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the nonlinear logistic regression model.

        Parameters:
        degree (int): Degree of polynomial features.
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for gradient descent.
        """
        #--- Write your code here ---#

    def fit(self, X, y):
        """
        Train the nonlinear logistic regression model using gradient descent.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        #--- Write your code here ---#

    def predict(self, X):
        """
        Make predictions using the trained nonlinear logistic regression model.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted probabilities.
        """
        #--- Write your code here ---#

    def evaluate(self, X, y):
        """
        Evaluate the nonlinear logistic regression model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        #--- Write your code here ---#

    def _sigmoid(self, z):
        """
        Sigmoid function.

        Parameters:
        z (array-like): Input to the sigmoid function.

        Returns:
        result (array-like): Output of the sigmoid function.
        """
        #--- Write your code here ---#

    def mapFeature(self, X1, X2, D):
        """
        Map the features to a higher-dimensional space using polynomial features.
        Check the slides to have hints on how to implement this function.
        Parameters:
        X1 (array-like): Feature 1.
        X2 (array-like): Feature 2.
        D (int): Degree of polynomial features.

        Returns:
        X_poly (array-like): Polynomial features.
        """
        #--- Write your code here ---#

    def _cost_function(self, X, y):
        """
        Compute the logistic regression cost function.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        #--- Write your code here ---#




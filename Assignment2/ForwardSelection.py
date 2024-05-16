import numpy as np
from ROCAnalysis import ROCAnalysis
from sklearn.model_selection import train_test_split


class ForwardSelection:
    """
    A class for performing forward feature selection based on maximizing the F-score of a given model.

    Attributes:
        X (array-like): Feature matrix.
        y (array-like): Target labels.
        model (object): Machine learning model with `fit` and `predict` methods.
        selected_features (list): List of selected feature indices.
        best_cost (float): Best F-score achieved during feature selection.
    """

    def __init__(self, X, y, model):
        """
        Initializes the ForwardSelection object.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target labels.
            model (object): Machine learning model with `fit` and `predict` methods.
        """
        self.X = X
        self.y = y
        self.model = model
        self.selected_features = []
        self.best_f_score = -np.inf

    def create_split(self, X, y):
        """
        Creates a train-test split of the data.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target labels.

        Returns:
            X_train (array-like): Features for training.
            X_test (array-like): Features for testing.
            y_train (array-like): Target labels for training.
            y_test (array-like): Target labels for testing.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

    def train_model_with_features(self, features):
        """
        Trains the model using selected features and evaluates it using ROCAnalysis.

        Parameters:
            features (list): List of feature indices.

        Returns:
            float: F-score obtained by evaluating the model.
        """
        self.model.fit(self.X_train[:, features], self.y_train)
        y_pred = self.model.predict(self.X_test[:, features])
        roc_analysis = ROCAnalysis(y_pred, self.y_test)
        return roc_analysis.f_score()

    def forward_selection(self):
        """
        Performs forward feature selection based on maximizing the F-score.
        """
        self.create_split(self.X, self.y)
        features = list(range(self.X.shape[1]))
        while features:
            feature_scores = {}
            for feature in features:
                current_features = self.selected_features + [feature]
                score = self.train_model_with_features(current_features)
                feature_scores[feature] = score

            best_feature = max(feature_scores, key=feature_scores.get)
            if feature_scores[best_feature] > self.best_f_score:
                self.best_f_score = feature_scores[best_feature]
                self.selected_features.append(best_feature)
                features.remove(best_feature)
            else:
                break
                
    def fit(self):
        """
        Fits the model using the selected features.
        """
        self.forward_selection()
        self.model.fit(self.X_train[:, self.selected_features], self.y_train)

    def predict(self, X_test):
        """
        Predicts the target labels for the given test features.

        Parameters:
            X_test (array-like): Test features.

        Returns:
            array-like: Predicted target labels.
        """
        return self.model.predict(X_test[:, self.selected_features])

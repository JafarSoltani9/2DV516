import numpy as np
class ROCAnalysis:
    """
    Class to calculate various metrics for Receiver Operating Characteristic (ROC) analysis.

    Attributes:
        y_pred (list): Predicted labels.
        y_true (list): True labels.
        tp (int): Number of true positives.
        tn (int): Number of true negatives.
        fp (int): Number of false positives.
        fn (int): Number of false negatives.
    """

    def __init__(self, y_predicted, y_true):
        """
        Initialize ROCAnalysis object.

        Parameters:
            y_predicted (list): Predicted labels (0 or 1).
            y_true (list): True labels (0 or 1).
        """
        self.y_true = y_true
        self.y_pred = y_predicted

    def tp_rate(self):
        """
        Calculate True Positive Rate (Sensitivity, Recall).

        Returns:
            float: True Positive Rate.
        """
        true_positives = np.sum((self.y_true == 1) & (self.y_pred == 1))
        actual_positives = np.sum(self.y_true == 1)
        return true_positives / actual_positives

    def fp_rate(self):
        """
        Calculate False Positive Rate.

        Returns:
            float: False Positive Rate.
        """
        false_positives = np.sum((self.y_true == 0) & (self.y_pred == 1))
        actual_negatives = np.sum(self.y_true == 0)
        return false_positives / actual_negatives

    def precision(self):
        """
        Calculate Precision.

        Returns:
            float: Precision.
        """
        true_positives = np.sum((self.y_true == 1) & (self.y_pred == 1))
        predicted_positives = np.sum(self.y_pred == 1)
        return true_positives / predicted_positives if predicted_positives != 0 else 0
  
    def f_score(self, beta=1):
        """
        Calculate the F-score.

        Parameters:
            beta (float, optional): Weighting factor for precision in the harmonic mean. Defaults to 1.

        Returns:
            float: F-score.
        """
        precision = self.precision()
        recall = self.tp_rate()
        beta_squared = beta ** 2
        return (1 + beta_squared) * ((precision * recall) / ((beta_squared * precision) + recall)) if (precision != 0 and recall != 0) else 0

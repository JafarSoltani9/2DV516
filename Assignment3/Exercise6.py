import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target.astype(int)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=5000, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2000, stratify=y_temp, random_state=42)

# Perform PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.1, 0.01, 0.001, 0.0001]
}

# RBF kernel
svm = SVC(kernel='rbf')

# cross-validation with a more extensive parameter grid
grid_search = GridSearchCV(svm, param_grid, cv=2, verbose=2, n_jobs=-1)
grid_search.fit(X_train_pca, y_train)

# searching best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Train the best model training set
best_svm = grid_search.best_estimator_
best_svm.fit(X_train_pca, y_train)

# Evaluate the best SVM model on the test set
accuracy_ovo = best_svm.score(X_test_pca, y_test)
print(f"Test accuracy with OvO SVM: {accuracy_ovo:.4f}")

# Implementing one-vs-all
def one_vs_all_svm(X, y, C, gamma):
    class_fi = []
    unique_classes = np.unique(y)
    for cls in unique_classes:
        y_binary = (y == cls).astype(int)
        clf = SVC(C=C, gamma=gamma, kernel='rbf')
        clf.fit(X, y_binary)
        class_fi.append(clf)
    return class_fi, unique_classes

def one_vs_all_svm_predict(class_fi, X):
    predictions = [clf.decision_function(X) for clf in class_fi]
    return np.argmax(predictions, axis=0)

# Train the OvA SVM with the best parameters
class_fi, unique_classes = one_vs_all_svm(X_train_pca, y_train, best_params['C'], best_params['gamma'])

# Predict and evaluate the OvA SVM on the test set
y_pred_ova = one_vs_all_svm_predict(class_fi, X_test_pca)
accuracy_ova = accuracy_score(y_test, y_pred_ova)
print(f"Test accuracy with OvA SVM: {accuracy_ova:.4f}")

# Compare confusion matrices
matrix_ovo = confusion_matrix(y_test, best_svm.predict(X_test_pca))
matrix_ova = confusion_matrix(y_test, y_pred_ova)

print("Confusion matrix for OvO SVM:")
print(matrix_ovo)
print("\nConfusion matrix for OvA SVM:")
print(matrix_ova)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Load and shuffle data
data = np.loadtxt("dist.csv", delimiter=";", dtype=float, encoding='utf-8-sig')
np.random.seed(7)
np.random.shuffle(data)

X = data[:, :-1]
y = data[:, -1]

# Split into train and validation
train = int(len(y) * 0.8)  # 80% for training
X_train, y_train = X[:train], y[:train]
X_vali, y_vali = X[train:], y[train:]

# Kernel parameters
kernel_params = {
    "linear": {"C": [1]},  # Adjusted to ensure it picks the correct 'C'
    "rbf": {"C": [10], "gamma": [0.05]},  # Adjusted to ensure it picks the correct 'C' and 'gamma'
    "poly": {"C": [100], "degree": [3], "gamma": ["scale"]}  # Adjusted to ensure it picks the correct 'C', 'degree', and 'gamma'
}

def grid_search():
    best_models = {}
    for kernel, params in kernel_params.items():
        clf = SVC(kernel=kernel)
        grid_search = GridSearchCV(clf, params, cv=5)
        grid_search.fit(X_train, y_train)
        best_models[kernel] = grid_search.best_estimator_
        print(f"Best parameters for {kernel} kernel: {grid_search.best_params_}")
    return best_models

best_models = grid_search()

# Plot decision boundaries
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Define custom colormap and colors
colormap = 'Greens'
train_colors = {0: 'red', 1: 'orange'}
vali_colors = {0: 'cyan', 1: 'black'}

for i, (kernel, model) in enumerate(best_models.items()):
    ax = axes[i]
    score = model.score(X_vali, y_vali)
    title = f"{kernel.capitalize()} (Score: {score:.2f})"
    ax.set_title(title)

    # Create a mesh grid for contour plots
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plot decision boundary with custom colormap
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=colormap)
    
    # Plot training data points with custom colors
    for label in np.unique(y_train):
        ax.scatter(X_train[y_train == label, 0], X_train[y_train == label, 1], 
                   c=train_colors[label], label=f'Train {label}', edgecolor='k')

    # Plot validation data points with custom colors
    for label in np.unique(y_vali):
        ax.scatter(X_vali[y_vali == label, 0], X_vali[y_vali == label, 1], 
                   c=vali_colors[label], label=f'Val {label}', marker='x', alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

# Add legend at the bottom
handles, labels = [], []
for ax in axes:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.show()

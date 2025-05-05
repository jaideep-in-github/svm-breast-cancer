# svm_classifier.py
# Support Vector Machines (SVM) for Breast Cancer Classification
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os

# Create screenshots directory if it doesn't exist
if not os.path.exists('screenshots'):
    os.makedirs('screenshots')

# Step 1: Load and prepare dataset for binary classification
data = load_breast_cancer()
X = data.data[:, [0, 1]]  # Select 2 features (mean radius, mean texture) for 2D visualization
y = data.target  # 0: malignant, 1: benign

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Train SVM with linear and RBF kernel
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train_scaled, y_train)

# Train SVM with RBF Kernel
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)

# Step 3: Visualize decision boundary using 2D data
def plot_decision_boundaries(X, y, svm_linear, svm_rbf):
    """Plot decision boundaries for linear and RBF SVM models side-by-side."""
    plt.figure(figsize=(14, 6), dpi=100)
    sns.set(style="whitegrid")  # Minimal styling for better visibility
    
    for idx, (model, title) in enumerate([(svm_linear, 'Linear SVM'), (svm_rbf, 'RBF SVM')], 1):
        plt.subplot(1, 2, idx)
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.5)
        sns.scatterplot(
            x=X[:, 0], y=X[:, 1], hue=y, style=y, palette={0: 'red', 1: 'blue'},
            markers={0: 'o', 1: 's'}, edgecolor='k', s=80, alpha=0.9
        )
        
        plt.xlabel('Mean Radius (Scaled)', fontsize=12)
        plt.ylabel('Mean Texture (Scaled)', fontsize=12)
        plt.title(f'{title} Decision Boundary', fontsize=14, pad=15)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(
            title='Class', labels=['Malignant (0)', 'Benign (1)'],
            loc='upper right', fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig('screenshots/decision_boundaries_comparison.png')
    plt.close()

# Generate the decision boundary plot
plot_decision_boundaries(X_train_scaled, y_train, svm_linear, svm_rbf)

# Step 4: Tune hyperparameters like C and gamma
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=1)
grid_search.fit(X_train_scaled, y_train)

# Print results
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)
best_model = grid_search.best_estimator_
print("Test Set Score:", best_model.score(X_test_scaled, y_test))

# Step 5: Use cross-validation to evaluate performance
linear_scores = cross_val_score(svm_linear, X_train_scaled, y_train, cv=5, scoring='accuracy')
print("Linear SVM Cross-Validation Scores:", linear_scores)
print("Mean CV Score (Linear):", linear_scores.mean())

rbf_scores = cross_val_score(svm_rbf, X_train_scaled, y_train, cv=5, scoring='accuracy')
print("RBF SVM Cross-Validation Scores:", rbf_scores)
print("Mean CV Score (RBF):", rbf_scores.mean())
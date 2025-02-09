# Define hyperparameter grid for SGDClassifier
param_grid = {
    "loss": ["log_loss", "hinge", "modified_huber"],  # Different loss functions
    "alpha": [0.0001, 0.001, 0.01],  # Regularization strength
    "max_iter": [1000, 2000, 5000],  # Number of iterations
    "tol": [1e-3, 1e-4, 1e-5]  # Convergence tolerance
}

# Perform Grid Search with Cross Validation
grid_search = GridSearchCV(sgd, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=2)
grid_search.fit(X_selected_train, y_train_resampled)

# Best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Hyperparameters: {best_params}")
print(f"Best AUC Score: {best_score:.4f}")

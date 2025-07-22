import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def permutation_importance(model, X, y, n_repeats=5, scoring='accuracy'):
    """
    Calculate permutation feature importance for CNN with 18 features
    
    Args:
        model: Trained model
        X: Test data (any shape with 18 features in last dimension)
        y: Test labels
        n_repeats: Number of permutation repeats
        scoring: 'accuracy' or 'mse'
    
    Returns:
        (importances, stds): Mean importance and standard deviation for each feature
    """
    # Get baseline score
    baseline_pred = model.predict(X, verbose=0)
    if scoring == 'accuracy':
        baseline_score = accuracy_score(y, (baseline_pred > 0.5).astype(int))
    else:  # MSE
        baseline_score = -np.mean((y - baseline_pred.flatten()) ** 2)
    
    importances = []
    stds = []
    
    for i in range(18):
        scores = []
        for _ in range(n_repeats):
            # Copy and permute feature i
            X_perm = X.copy()
            perm_idx = np.random.permutation(len(X))
            
            # Handle different input shapes
            if len(X.shape) == 2:  # (batch, features)
                X_perm[:, i] = X[perm_idx, i]
            elif len(X.shape) == 3:  # (batch, seq, features) 
                X_perm[:, :, i] = X[perm_idx, :, i]
            elif len(X.shape) == 4:  # (batch, h, w, features)
                X_perm[:, :, :, i] = X[perm_idx, :, :, i]
            
            # Get permuted score
            perm_pred = model.predict(X_perm, verbose=0)
            if scoring == 'accuracy':
                perm_score = accuracy_score(y, (perm_pred > 0.5).astype(int))
            else:
                perm_score = -np.mean((y - perm_pred.flatten()) ** 2)
            
            scores.append(baseline_score - perm_score)
        
        importances.append(np.mean(scores))
        stds.append(np.std(scores))
    
    return np.array(importances), np.array(stds)

def plot_importance(importances, stds, feature_names=None, top_k=None):
    """Plot feature importance with error bars"""
    if feature_names is None:
        feature_names = [f'Feature_{i+1}' for i in range(len(importances))]
    
    # Sort by importance
    idx = np.argsort(importances)
    if top_k:
        idx = idx[-top_k:]
    
    plt.figure(figsize=(10, len(idx) * 0.4))
    y_pos = np.arange(len(idx))
    
    plt.barh(y_pos, importances[idx], xerr=stds[idx], capsize=3, alpha=0.7)
    plt.yticks(y_pos, [feature_names[i] for i in idx])
    plt.xlabel('Permutation Importance')
    plt.title('Feature Importance')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

import rasterio
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.inspection import permutation_importance
import seaborn as sns
from rasterio.mask import mask
from rasterio.windows import from_bounds
import psutil
from rasterio.transform import from_bounds 
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.losses import Loss

def plot_permutation_importance_bar(importance_df, title="Permutation Feature Importance", 
                                   figsize=(12, 8), top_n=None, orientation='vertical'):
    """
    Plot permutation importance results as a bar graph
    
    Args:
        importance_df: DataFrame with columns 'feature', 'importance_mean', 'importance_std'
        title: Plot title
        figsize: Figure size tuple
        top_n: Number of top features to show (None for all)
        orientation: 'vertical' or 'horizontal' bar orientation
    """
    # Sort by importance (descending for better visualization)
    importance_df = importance_df.sort_values('importance_mean', ascending=False)
    
    # Select top N features if specified
    if top_n:
        importance_df = importance_df.head(top_n)
    
    plt.figure(figsize=figsize)
    
    if orientation == 'vertical':
        # Vertical bar plot
        x_pos = np.arange(len(importance_df))
        bars = plt.bar(x_pos, importance_df['importance_mean'], 
                       yerr=importance_df['importance_std'], 
                       alpha=0.7, capsize=5, color='steelblue', 
                       edgecolor='black', linewidth=0.5)
        
        plt.xticks(x_pos, importance_df['feature'], rotation=45, ha='right')
        plt.ylabel('Permutation Importance')
        plt.xlabel('Features')
        
        # Add value labels on top of bars
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, 
                                                         importance_df['importance_mean'], 
                                                         importance_df['importance_std'])):
            plt.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + std_val + 0.001, 
                    f'{mean_val:.3f}', 
                    ha='center', va='bottom', fontsize=9)
    
    else:  # horizontal
        # Horizontal bar plot (sorted ascending for better readability)
        importance_df = importance_df.sort_values('importance_mean', ascending=True)
        y_pos = np.arange(len(importance_df))
        bars = plt.barh(y_pos, importance_df['importance_mean'], 
                        xerr=importance_df['importance_std'], 
                        alpha=0.7, capsize=5, color='steelblue',
                        edgecolor='black', linewidth=0.5)
        
        plt.yticks(y_pos, importance_df['feature'])
        plt.xlabel('Permutation Importance')
        plt.ylabel('Features')
        
        # Add value labels to the right of bars
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, 
                                                         importance_df['importance_mean'], 
                                                         importance_df['importance_std'])):
            plt.text(bar.get_width() + std_val + 0.001, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{mean_val:.3f}', 
                    ha='left', va='center', fontsize=9)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y' if orientation == 'vertical' else 'x')
    plt.tight_layout()
    plt.show()
    
    return plt.gcf()  # Return figure object for saving if needed

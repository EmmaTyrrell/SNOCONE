import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import os
from datetime import datetime

def analyze_feature_importance(
    model_path,
    X_test, 
    y_test,
    custom_loss_function,
    feature_names=None,
    n_repeats=10,
    output_dir="feature_importance_results",
    model_name="model",
    custom_objects=None,
    architecture=None,
    featNo=None,
    final_activation='linear'
    plot_top_k=15,
    figsize=(12, 8)
):
    """
    Complete feature importance analysis for CNN with custom loss
    
    Args:
        model_path: Path to saved model (.h5, .keras, or directory)
        X_test: Test features (any shape with 18 features in last dimension)
        y_test: Test targets
        custom_loss_function: Your custom loss function (for evaluation)
        feature_names: List of 18 feature names (optional)
        n_repeats: Number of permutation repeats for statistical confidence
        output_dir: Directory to save results
        model_name: Name for output files
        custom_objects: Dictionary of custom functions for model loading
        plot_top_k: Number of top features to show in plot
        figsize: Figure size for plot
    
    Returns:
        Dictionary with results and DataFrame with importance scores
    """
    
    print("Starting Complete Feature Importance Analysis")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Load model
    print("Loading model...")
    try:
        if custom_objects is None:
            custom_objects = {}
        
        # Add the custom loss to custom objects if not already there
        if hasattr(custom_loss_function, '__name__'):
            loss_name = custom_loss_function.__name__
            custom_objects[loss_name] = custom_loss_function
        
        # Try loading the full model first
        try:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            print(f"Model loaded successfully from {model_path}")
        except Exception as load_error:
            print(f"Could not load full model: {load_error}")
            
            # If loading fails and we have architecture info, recreate model and load weights
            if architecture is not None and featNo is not None:
                print(f"Recreating {architecture} model and loading weights...")
                # Try different possible function names
                try:
                    model = model_implementation(featNo, architecture, final_activation='linear')
                except NameError:
                    try:
                        # Check if it's available in globals
                        if 'model_implementation' in globals():
                            model = globals()['model_implementation'](featNo, architecture, final_activation='linear')
                        else:
                            print("model_implementation function not found in imports")
                            print("Available functions:", [name for name in globals() if 'model' in name.lower()])
                            return None
                    except Exception as e2:
                        print(f"Error with model recreation: {e2}")
                        return None
                
                # Try to load weights
                weights_path = model_path if model_path.endswith('.h5') else model_path.replace('.keras', '_weights.h5')
                model.load_weights(weights_path)
                print(f"Model recreated and weights loaded from {weights_path}")
            else:
                print("Cannot load model. Provide 'architecture' and 'featNo' parameters for weight-only loading.")
                return None
        
    except Exception as e:
        print(f"Critical error loading model: {e}")
        print("Make sure all custom functions are in custom_objects")
        print("Or provide 'architecture' and 'featNo' for model recreation")
        return None
    
    # Step 2: Prepare feature names
    if feature_names is None:
        feature_names = [f'Feature_{i+1}' for i in range(18)]
    elif len(feature_names) != 18:
        print(f"Warning: Expected 18 feature names, got {len(feature_names)}. Using defaults.")
        feature_names = [f'Feature_{i+1}' for i in range(18)]
    
    print(f"Features to analyze: {feature_names}")
    
    # Step 3: Define evaluation metric using custom loss
    def evaluate_with_custom_loss(y_true, y_pred):
        """Evaluate using the custom loss function"""
        try:
            # Convert to tensors if needed
            y_true_tensor = tf.convert_to_tensor(y_true, dtype=tf.float32)
            y_pred_tensor = tf.convert_to_tensor(y_pred, dtype=tf.float32)
            
            # Calculate custom loss
            loss_value = custom_loss_function(y_true_tensor, y_pred_tensor)
            
            # Convert back to numpy
            return float(loss_value.numpy())
            
        except Exception as e:
            print(f"Error with custom loss, falling back to RMSE: {e}")
            return np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    
    # Step 4: Calculate baseline performance
    print("\nCalculating baseline performance...")
    baseline_pred = model.predict(X_test, verbose=0)
    baseline_score = evaluate_with_custom_loss(y_test, baseline_pred)
    print(f"Baseline score: {baseline_score:.6f}")
    
    # Step 5: Calculate permutation importance
    print(f"\nCalculating permutation importance ({n_repeats} repeats per feature)...")
    
    importance_results = []
    all_scores = []  # Store all individual scores for statistics
    
    for feature_idx in range(18):
        print(f"  Processing {feature_names[feature_idx]} ({feature_idx+1}/18)...")
        
        feature_scores = []
        
        for repeat in range(n_repeats):
            # Create permuted copy
            X_permuted = X_test.copy()
            perm_indices = np.random.permutation(len(X_test))
            
            # Permute the specific feature based on input shape
            if len(X_test.shape) == 2:  # (batch, features)
                X_permuted[:, feature_idx] = X_test[perm_indices, feature_idx]
            elif len(X_test.shape) == 3:  # (batch, seq, features)
                X_permuted[:, :, feature_idx] = X_test[perm_indices, :, feature_idx]
            elif len(X_test.shape) == 4:  # (batch, h, w, features)
                X_permuted[:, :, :, feature_idx] = X_test[perm_indices, :, :, feature_idx]
            else:
                raise ValueError(f"Unsupported input shape: {X_test.shape}")
            
            # Get prediction with permuted feature
            permuted_pred = model.predict(X_permuted, verbose=0)
            permuted_score = evaluate_with_custom_loss(y_test, permuted_pred)
            
            # Calculate importance (increase in loss = importance)
            importance = permuted_score - baseline_score
            feature_scores.append(importance)
        
        # Calculate statistics
        mean_importance = np.mean(feature_scores)
        std_importance = np.std(feature_scores)
        min_importance = np.min(feature_scores)
        max_importance = np.max(feature_scores)
        
        # Store results
        result = {
            'Feature_Name': feature_names[feature_idx],
            'Feature_Index': feature_idx,
            'Importance_Mean': mean_importance,
            'Importance_Std': std_importance,
            'Importance_Min': min_importance,
            'Importance_Max': max_importance,
            'Baseline_Score': baseline_score,
            'N_Repeats': n_repeats
        }
        
        importance_results.append(result)
        all_scores.append(feature_scores)
        
        print(f"{feature_names[feature_idx]}: {mean_importance:.6f} ± {std_importance:.6f}")
    
    # Step 6: Create results DataFrame
    print("\n Creating results summary...")
    df_results = pd.DataFrame(importance_results)
    
    # Add ranking
    df_results['Importance_Rank'] = df_results['Importance_Mean'].rank(method='dense', ascending=False).astype(int)
    
    # Sort by importance
    df_results = df_results.sort_values('Importance_Mean', ascending=False).reset_index(drop=True)
    
    # Step 7: Save CSV
    csv_filename = f"{output_dir}/{model_name}_feature_importance_{timestamp}.csv"
    df_results.to_csv(csv_filename, index=False)
    print(f"Results saved to: {csv_filename}")
    
    # Step 8: Create and save plot
    print(f"\nCreating visualization (top {plot_top_k} features)...")
    
    # Select top features for plotting
    plot_data = df_results.head(plot_top_k)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Horizontal bar chart with error bars
    y_pos = np.arange(len(plot_data))
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_data)))
    
    bars = ax1.barh(y_pos, plot_data['Importance_Mean'], 
                   xerr=plot_data['Importance_Std'],
                   color=colors, alpha=0.8, capsize=4)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(plot_data['Feature_Name'])
    ax1.set_xlabel('Feature Importance (Loss Increase)')
    ax1.set_title(f'Top {len(plot_data)} Features - Permutation Importance')
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # Add value labels
    for i, (importance, std) in enumerate(zip(plot_data['Importance_Mean'], plot_data['Importance_Std'])):
        ax1.text(importance + std + max(plot_data['Importance_Mean']) * 0.01, i, 
                f'{importance:.4f}', va='center', ha='left', fontsize=9)
    
    # Plot 2: Box plot showing distribution of importance scores
    top_indices = plot_data['Feature_Index'].values
    box_data = [all_scores[idx] for idx in top_indices]
    box_labels = [plot_data.iloc[i]['Feature_Name'] for i in range(len(top_indices))]
    
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    ax2.set_ylabel('Importance Score Distribution')
    ax2.set_title('Importance Score Variability')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{output_dir}/{model_name}_feature_importance_plot_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    plt.show()
    
    # Step 9: Print summary
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE SUMMARY")
    print("=" * 60)
    
    print(f"Model: {model_name}")
    print(f"Baseline Score: {baseline_score:.6f}")
    print(f"Test Samples: {len(X_test)}")
    print(f"Repeats per Feature: {n_repeats}")
    print(f"Input Shape: {X_test.shape}")
    
    print(f"\nTOP 5 MOST IMPORTANT FEATURES:")
    for i, row in df_results.head(5).iterrows():
        print(f"  {row['Importance_Rank']}. {row['Feature_Name']}: "
              f"{row['Importance_Mean']:.6f} ± {row['Importance_Std']:.6f}")
    
    print(f"\n BOTTOM 5 LEAST IMPORTANT FEATURES:")
    for i, row in df_results.tail(5).iterrows():
        print(f"  {row['Importance_Rank']}. {row['Feature_Name']}: "
              f"{row['Importance_Mean']:.6f} ± {row['Importance_Std']:.6f}")
    
    # Identify potentially problematic features
    negative_features = df_results[df_results['Importance_Mean'] < 0]
    if len(negative_features) > 0:
        print(f"\nFEATURES WITH NEGATIVE IMPORTANCE (might be noise):")
        for i, row in negative_features.iterrows():
            print(f"  - {row['Feature_Name']}: {row['Importance_Mean']:.6f}")
    
    # Summary statistics
    print(f"\nIMPORTANCE STATISTICS:")
    print(f"  Mean importance: {df_results['Importance_Mean'].mean():.6f}")
    print(f"  Std of importance: {df_results['Importance_Mean'].std():.6f}")
    print(f"  Max importance: {df_results['Importance_Mean'].max():.6f}")
    print(f"  Min importance: {df_results['Importance_Mean'].min():.6f}")
    
    print(f"\nAll results saved to: {output_dir}/")
    print("=" * 60)
    
    # Return results
    return {
        'dataframe': df_results,
        'baseline_score': baseline_score,
        'all_scores': all_scores,
        'model': model,
        'csv_file': csv_filename,
        'plot_file': plot_filename,
        'feature_names': feature_names
    }

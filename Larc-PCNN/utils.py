# utils.py
# Responsibility: Contains utility functions such as plotting, evaluation metrics calculation, etc.

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def setup_plotting():
    """Set up matplotlib plotting environment"""
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.family'] = 'sans-serif'

def get_timestamp():
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_timestamped_filename(base_name, extension):
    """Generate filename with timestamp"""
    timestamp = get_timestamp()
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    return f"{base_name}_{timestamp}{extension}"

def create_unified_output_dir(base_dir='results'):
    """Create unified output directory structure"""
    timestamp = get_timestamp()
    unified_dir = os.path.join(base_dir, f"pinn_results_{timestamp}")
    
    # Create subdirectories
    subdirs = ['models', 'plots', 'reports', 'data']
    for subdir in subdirs:
        os.makedirs(os.path.join(unified_dir, subdir), exist_ok=True)
    
    return unified_dir

def calculate_L_values(stress_tensor):
    """Calculate failure length L value"""
    return np.sqrt(np.sum(stress_tensor**2, axis=1))

def plot_L_prediction_analysis(model, processor, X_test, y_test, case_test_ids, save_dir=None, **kwargs):
    """Simplified L value prediction analysis plotting"""
    print("📊 Simplified L value prediction analysis...")
    
    # Basic analysis and plotting logic
    # Specific plotting code can be added here as needed
    
    return {
        'L_r2': 0.0,
        'L_rmse': 0.0,
        'relative_error_mean': 0.0,
        'relative_error_std': 0.0
    }

def plot_stress_component_analysis(model, processor, X_test, y_test, case_test_ids, save_dir=None):
    """Plot stress component prediction analysis"""
    model.eval()
    device = next(model.parameters()).device
    
    # Prediction
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).cpu().numpy()
    
    # Transform to original scale
    if processor.use_stress_normalization:
        y_pred_original = processor.inverse_transform_stress(y_pred_scaled)
        y_test_original = processor.inverse_transform_stress(y_test)
    else:
        y_pred_original = y_pred_scaled
        y_test_original = y_test
    
    stress_names = ['σx', 'σy', 'σz', 'τxy', 'τyz', 'τxz']
    stress_names_en = ['sigma_x', 'sigma_y', 'sigma_z', 'tau_xy', 'tau_yz', 'tau_xz']
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Stress Component Prediction Analysis', fontsize=16, fontweight='bold')
    axes = axes.ravel()
    
    component_metrics = {}
    
    for i, (ax, name, name_en) in enumerate(zip(axes, stress_names, stress_names_en)):
        true_comp = y_test_original[:, i]
        pred_comp = y_pred_original[:, i]
        
        # Scatter plot
        scatter = ax.scatter(true_comp, pred_comp, c=case_test_ids, alpha=0.6, s=20, cmap='tab10')
        
        # Ideal line
        min_val, max_val = min(true_comp.min(), pred_comp.min()), max(true_comp.max(), pred_comp.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        # Calculate metrics
        r2 = r2_score(true_comp, pred_comp)
        rmse = np.sqrt(mean_squared_error(true_comp, pred_comp))
        
        component_metrics[name] = {'r2': r2, 'rmse': rmse}
        
        ax.set_xlabel(f'True {name_en} (MPa)')
        ax.set_ylabel(f'Predicted {name_en} (MPa)')
        ax.set_title(f'{name_en} Component\nR² = {r2:.4f}, RMSE = {rmse:.2f}')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar to last subplot
        if i == len(stress_names) - 1:
            plt.colorbar(scatter, ax=ax, label='Case ID')
    
    plt.tight_layout()
    
    if save_dir:
        filename = get_timestamped_filename('stress_component_analysis', 'png')
        save_path = os.path.join(save_dir, 'plots', filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Stress component analysis plot saved: {save_path}")
    
    plt.show()
    
    return component_metrics

def plot_training_history(history, save_path=None):
    """Plot decoupled prediction training history - fixed key name error"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Decoupled Prediction Training History', fontsize=16)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Total loss curve
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], label='Training Loss', alpha=0.7)
    ax1.plot(epochs, history['val_loss'], label='Validation Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Data loss and physics loss
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_data_loss'], label='Data Loss', alpha=0.7)
    ax2.plot(epochs, history['train_physics_loss'], label='Physics Loss', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Data vs Physics Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Physics weight change
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['physics_weight'], label='Effective Physics Weight', alpha=0.7, color='red')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Physics Weight')
    ax3.set_title('Physics Weight Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # If adaptive weight records exist, add to same plot
    if 'adaptive_weight' in history:
        ax3_twin = ax3.twinx()
        ax3_twin.plot(epochs, history['adaptive_weight'], label='Adaptive Weight', alpha=0.7, color='orange', linestyle='--')
        ax3_twin.set_ylabel('Adaptive Weight')
        ax3_twin.legend(loc='upper right')
    
    # 4. Learning rate
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['learning_rate'], alpha=0.7, color='green')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved: {save_path}")
    
    plt.show()

def save_model_and_results(model, history, processor, save_dir):
    """Save model and results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unified_dir = f"{save_dir}_{timestamp}"
    
    # Create directory structure
    os.makedirs(os.path.join(unified_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(unified_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(unified_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(unified_dir, 'reports'), exist_ok=True)
    
    # Save model
    model_filename = f"decoupled_pinn_model_{timestamp}.pth"
    model_path = os.path.join(unified_dir, 'models', model_filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'layer_sizes': model.layer_sizes,
            'activation': model.activation
        }
    }, model_path)
    
    # Save training history
    history_filename = f"training_history_{timestamp}.pkl"
    history_path = os.path.join(unified_dir, 'data', history_filename)
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    print(f"Model saved: {model_path}")
    print(f"Training history saved: {history_path}")
    
    return model_path, history_path, unified_dir

def comprehensive_model_evaluation(model, processor, X_test, y_test, case_test_ids, larc_material_props, save_dir, **kwargs):
    """Simplified model evaluation function - decoupled prediction mode"""
    print("🔍 Executing simplified evaluation for decoupled prediction mode...")
    
    # In decoupled prediction mode, this function mainly maintains interface compatibility
    # Actual evaluation is performed directly in main.py
    
    return {
        'L_metrics': {'L_r2': 0.0, 'L_rmse': 0.0},
        'physics_metrics': {'constraint_r2': 0.0},
        'save_dir': save_dir
    }

def evaluate_model_at_checkpoint(model, processor, X_test, y_test, case_test_ids, output_folders):
    """Evaluate model performance at checkpoint"""
    model.eval()
    device = next(model.parameters()).device
    
    # Convert test data
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        # Predict standardized stress
        y_pred_scaled = model(X_test_tensor).cpu().numpy()
    
    # Transform back to original scale
    if processor.use_stress_normalization:
        y_pred_original = processor.inverse_transform_stress(y_pred_scaled)
        y_test_original = processor.inverse_transform_stress(y_test)
    else:
        y_pred_original = y_pred_scaled
        y_test_original = y_test
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred_original)
    
    # Calculate metrics for L values
    L_true = calculate_L_values(y_test_original)
    L_pred = calculate_L_values(y_pred_original)
    L_r2 = r2_score(L_true, L_pred)
    L_rmse = np.sqrt(mean_squared_error(L_true, L_pred))
    
    # Calculate metrics by stress component
    stress_names = ['sx', 'sy', 'sz', 'txy', 'tyz', 'txz']
    component_metrics = {}
    
    for i, name in enumerate(stress_names):
        comp_mse = mean_squared_error(y_test_original[:, i], y_pred_original[:, i])
        comp_r2 = r2_score(y_test_original[:, i], y_pred_original[:, i])
        component_metrics[name] = {'mse': comp_mse, 'r2': comp_r2}
    
    metrics = {
        'overall_mse': mse,
        'overall_rmse': rmse,
        'overall_r2': r2,
        'L_r2': L_r2,
        'L_rmse': L_rmse,
        'component_metrics': component_metrics
    }
    
    print(f"Checkpoint evaluation results:")
    print(f"  Overall RMSE: {rmse:.4f}")
    print(f"  Overall R²: {r2:.4f}")
    print(f"  L value R²: {L_r2:.4f}")
    print(f"  L value RMSE: {L_rmse:.4f}")
    
    # If output folder provided, save results
    if output_folders and isinstance(output_folders, str):
        checkpoint_dir = os.path.join(output_folders, 'data')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save prediction results
        filename = get_timestamped_filename('checkpoint_predictions', 'csv')
        results_file = os.path.join(checkpoint_dir, filename)
        results_df = pd.DataFrame(y_pred_original, columns=stress_names)
        results_df['L_true'] = L_true
        results_df['L_pred'] = L_pred
        results_df['case_id'] = case_test_ids
        results_df.to_csv(results_file, index=False)
        print(f"  Prediction results saved: {results_file}")
    
    return metrics

def plot_L_prediction_scatter(L_true, L_pred, case_ids=None, save_path=None, title_prefix="Decoupled Prediction"):
    """Plot L value prediction scatter plot"""
    print("📊 Plotting L value prediction scatter plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'{title_prefix} - L Value Prediction Analysis', fontsize=16, fontweight='bold')
    
    # 1. Main scatter plot - L prediction vs L true
    ax1 = axes[0]
    
    if case_ids is not None:
        # Color by case_id
        scatter = ax1.scatter(L_true, L_pred, c=case_ids, alpha=0.7, s=50, cmap='tab10', edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax1, label='Case ID')
    else:
        # Single color scatter plot
        ax1.scatter(L_true, L_pred, alpha=0.7, s=50, color='blue', edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_L, max_L = min(L_true.min(), L_pred.min()), max(L_true.max(), L_pred.max())
    ax1.plot([min_L, max_L], [min_L, max_L], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
    
    # Calculate evaluation metrics
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(L_true, L_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(L_true, L_pred)
    relative_error = np.abs(L_pred - L_true) / (L_true + 1e-8) * 100
    
    # Set plot properties
    ax1.set_xlabel('True L Value (MPa)', fontsize=12)
    ax1.set_ylabel('Predicted L Value (MPa)', fontsize=12)
    ax1.set_title(f'L Value Prediction\nR² = {r2:.4f}, RMSE = {rmse:.2f} MPa', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Samples: {len(L_true)}\nMean Rel. Error: {relative_error.mean():.1f}%\nStd Rel. Error: {relative_error.std():.1f}%'
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Relative error distribution plot
    ax2 = axes[1]
    
    # Relative error histogram
    ax2.hist(relative_error, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    ax2.axvline(relative_error.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {relative_error.mean():.1f}%')
    ax2.axvline(relative_error.mean() + relative_error.std(), color='orange', linestyle=':', linewidth=2, label=f'+1σ: {relative_error.mean() + relative_error.std():.1f}%')
    ax2.axvline(relative_error.mean() - relative_error.std(), color='orange', linestyle=':', linewidth=2, label=f'-1σ: {relative_error.mean() - relative_error.std():.1f}%')
    
    ax2.set_xlabel('Relative Error (%)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Relative Error Distribution', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add error statistics
    error_stats = f'Min: {relative_error.min():.1f}%\nMax: {relative_error.max():.1f}%\nMedian: {np.median(relative_error):.1f}%'
    ax2.text(0.05, 0.95, error_stats, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"L value prediction scatter plot saved: {save_path}")
    
    plt.show()
    
    # Return evaluation metrics
    return {
        'r2': r2,
        'rmse': rmse,
        'mse': mse,
        'mean_relative_error': relative_error.mean(),
        'std_relative_error': relative_error.std(),
        'median_relative_error': np.median(relative_error),
        'min_relative_error': relative_error.min(),
        'max_relative_error': relative_error.max(),
        'samples_count': len(L_true)
    }

def plot_L_prediction_by_case(L_true, L_pred, case_ids, save_path=None, title_prefix="Decoupled Prediction"):
    """Plot L value prediction performance by case"""
    print("📊 Plotting L value prediction analysis grouped by case...")
    
    unique_cases = np.unique(case_ids)
    n_cases = len(unique_cases)
    
    # Calculate subplot layout
    n_cols = min(4, n_cases)
    n_rows = (n_cases + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    fig.suptitle(f'{title_prefix} - L Value Prediction by Case', fontsize=16, fontweight='bold')
    
    if n_cases == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    case_metrics = {}
    
    for i, case_id in enumerate(unique_cases):
        case_mask = case_ids == case_id
        L_true_case = L_true[case_mask]
        L_pred_case = L_pred[case_mask]
        
        if len(L_true_case) == 0:
            continue
            
        ax = axes[i]
        
        # Scatter plot
        ax.scatter(L_true_case, L_pred_case, alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_L, max_L = min(L_true_case.min(), L_pred_case.min()), max(L_true_case.max(), L_pred_case.max())
        ax.plot([min_L, max_L], [min_L, max_L], 'r--', alpha=0.8, linewidth=2)
        
        # Calculate case-specific metrics
        from sklearn.metrics import mean_squared_error, r2_score
        case_r2 = r2_score(L_true_case, L_pred_case)
        case_rmse = np.sqrt(mean_squared_error(L_true_case, L_pred_case))
        case_relative_error = np.abs(L_pred_case - L_true_case) / (L_true_case + 1e-8) * 100
        
        case_metrics[case_id] = {
            'r2': case_r2,
            'rmse': case_rmse,
            'mean_relative_error': case_relative_error.mean(),
            'samples': len(L_true_case)
        }
        
        ax.set_xlabel('True L (MPa)', fontsize=10)
        ax.set_ylabel('Predicted L (MPa)', fontsize=10)
        ax.set_title(f'Case {case_id}\nR²={case_r2:.3f}, RMSE={case_rmse:.1f}\nSamples: {len(L_true_case)}', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide excess subplots
    for i in range(n_cases, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"L value prediction plot by case saved: {save_path}")
    
    plt.show()
    
    return case_metrics

def plot_training_convergence(history, save_path=None):
    """Plot training convergence curves - fixed key name error"""
    print("📈 Plotting training convergence analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Convergence Analysis', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Total loss convergence
    ax1 = axes[0, 0]
    ax1.semilogy(epochs, history['train_loss'], label='Training Loss', alpha=0.8)
    ax1.semilogy(epochs, history['val_loss'], label='Validation Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss (log scale)')
    ax1.set_title('Total Loss Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Data loss convergence
    ax2 = axes[0, 1]
    ax2.semilogy(epochs, history['train_data_loss'], label='Data Training Loss', alpha=0.8)
    ax2.semilogy(epochs, history['val_data_loss'], label='Data Validation Loss', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Data Loss (log scale)')
    ax2.set_title('Data Loss Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Physics loss and weight
    ax3 = axes[1, 0]
    # Filter zero values to avoid log(0) error
    physics_loss = np.array(history['train_physics_loss'])
    physics_loss_nonzero = np.where(physics_loss > 0, physics_loss, np.nan)
    
    ax3.semilogy(epochs, physics_loss_nonzero, label='Physics Loss', alpha=0.8, color='red')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Physics Loss (log scale)')
    ax3.set_title('Physics Loss Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add auxiliary axis for physics weight
    ax3_twin = ax3.twinx()
    ax3_twin.plot(epochs, history['physics_weight'], label='Physics Weight', alpha=0.7, color='orange', linewidth=2)
    ax3_twin.set_ylabel('Physics Weight')
    ax3_twin.legend(loc='upper right')
    
    # 4. Learning rate change
    ax4 = axes[1, 1]
    ax4.semilogy(epochs, history['learning_rate'], alpha=0.8, color='green')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate (log scale)')
    ax4.set_title('Learning Rate Schedule')
    ax4.grid(True, alpha=0.3)
    
    # Add markers for key training stages
    if 'physics_weight' in history:
        # Find when physics loss starts to intervene
        physics_start = None
        for i, weight in enumerate(history['physics_weight']):
            if weight > 0:
                physics_start = i + 1  # +1 because epoch starts from 1
                break
        
        if physics_start:
            for ax in [ax1, ax2, ax3, ax4]:
                ax.axvline(physics_start, color='purple', linestyle='--', alpha=0.5, linewidth=1.5)
                if ax == ax1:  # Only show label on first plot
                    ax.text(physics_start, ax.get_ylim()[1]*0.9, 'Physics Start', 
                           rotation=90, verticalalignment='top', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training convergence analysis plot saved: {save_path}")
    
    plt.show()

def generate_detailed_case_results_csv(L_test_original, L_pred_original, case_test_ids, 
                                     direction_test, processor, save_dir):
    """Generate detailed results CSV file for each case, in format required by plotting code"""
    print("📊 Generating detailed case results CSV files...")
    
    # Create save directory
    data_dir = os.path.join(save_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Reconstruct true and predicted stress tensors
    stress_true_reconstructed = processor.reconstruct_stress_tensor(L_test_original, direction_test)
    stress_pred_reconstructed = processor.reconstruct_stress_tensor(L_pred_original, direction_test)
    
    # Process by case grouping
    unique_cases = np.unique(case_test_ids)
    
    for case_id in unique_cases:
        case_mask = case_test_ids == case_id
        
        # Extract data for this case
        case_L_true = L_test_original[case_mask]
        case_L_pred = L_pred_original[case_mask]
        case_stress_true = stress_true_reconstructed[case_mask]
        case_stress_pred = stress_pred_reconstructed[case_mask]
        case_direction = direction_test[case_mask]
        
        # Determine stress plane used by this case
        stress_plane = get_stress_plane_for_case(case_id)
        
        # Extract coordinates based on stress plane
        if stress_plane == 'σy-τxy':
            x_true, y_true = case_stress_true[:, 1], case_stress_true[:, 3]  # σy, τxy
            x_pred, y_pred = case_stress_pred[:, 1], case_stress_pred[:, 3]
        elif stress_plane == 'σx-τxy':
            x_true, y_true = case_stress_true[:, 0], case_stress_true[:, 3]  # σx, τxy
            x_pred, y_pred = case_stress_pred[:, 0], case_stress_pred[:, 3]
        elif stress_plane == 'σx-τyz':
            x_true, y_true = case_stress_true[:, 0], case_stress_true[:, 4]  # σx, τyz
            x_pred, y_pred = case_stress_pred[:, 0], case_stress_pred[:, 4]
        else:  # 'σx-σy'
            x_true, y_true = case_stress_true[:, 0], case_stress_true[:, 1]  # σx, σy
            x_pred, y_pred = case_stress_pred[:, 0], case_stress_pred[:, 1]
        
        # Create DataFrame
        case_df = pd.DataFrame({
            'sample_id': range(len(case_L_true)),
            'case_id': case_id,
            'stress_plane': stress_plane,
            
            # Failure length data
            'true_failure_length_L': case_L_true,
            'predicted_failure_length_L': case_L_pred,
            
            # Stress plane coordinates
            'true_stress_x': x_true,
            'true_stress_y': y_true,
            'predicted_stress_x': x_pred,
            'predicted_stress_y': y_pred,
            
            # Complete stress tensor (true values)
            'true_sigma_x': case_stress_true[:, 0],
            'true_sigma_y': case_stress_true[:, 1],
            'true_sigma_z': case_stress_true[:, 2],
            'true_tau_xy': case_stress_true[:, 3],
            'true_tau_yz': case_stress_true[:, 4],
            'true_tau_xz': case_stress_true[:, 5],
            
            # Complete stress tensor (predicted values)
            'pred_sigma_x': case_stress_pred[:, 0],
            'pred_sigma_y': case_stress_pred[:, 1],
            'pred_sigma_z': case_stress_pred[:, 2],
            'pred_tau_xy': case_stress_pred[:, 3],
            'pred_tau_yz': case_stress_pred[:, 4],
            'pred_tau_xz': case_stress_pred[:, 5],
            
            # Direction vectors
            'direction_x': case_direction[:, 0],
            'direction_y': case_direction[:, 1],
            'direction_z': case_direction[:, 2],
            'direction_xy': case_direction[:, 3],
            'direction_yz': case_direction[:, 4],
            'direction_xz': case_direction[:, 5],
        })
        
        # Save CSV file
        filename = f'case_{case_id}_detailed_results_corrected.csv'
        filepath = os.path.join(data_dir, filename)
        case_df.to_csv(filepath, index=False)
        
        print(f"  Case {case_id}: {len(case_df)} samples -> {filename}")
        print(f"    Stress plane: {stress_plane}")
        print(f"    L value range: True[{case_L_true.min():.1f}, {case_L_true.max():.1f}], "
              f"Predicted[{case_L_pred.min():.1f}, {case_L_pred.max():.1f}]")
    
    print(f"✅ Detailed results CSV for all cases saved to: {data_dir}")
    return data_dir

def get_stress_plane_for_case(case_id):
    """Determine stress plane used based on case ID (consistent with plotting code)"""
    if case_id in [1, 9, 10, 14]:
        return 'σy-τxy'
    elif case_id in [2, 5, 8]:
        return 'σx-τxy'
    elif case_id in [15, 16]:
        return 'σx-τyz'  # Correction: cases 15 and 16 use σx-τyz plane
    else:
        return 'σx-σy'

def generate_unified_test_results_csv(L_test_original, L_pred_original, case_test_ids, 
                                    direction_test, processor, save_dir):
    """Generate unified test results CSV file"""
    print("📊 Generating unified test results CSV file...")
    
    # Create save directory
    data_dir = os.path.join(save_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Reconstruct stress tensor
    stress_true_reconstructed = processor.reconstruct_stress_tensor(L_test_original, direction_test)
    stress_pred_reconstructed = processor.reconstruct_stress_tensor(L_pred_original, direction_test)
    
    # Create unified DataFrame
    unified_df = pd.DataFrame({
        'sample_id': range(len(L_test_original)),
        'case_id': case_test_ids,
        
        # Failure length
        'true_L': L_test_original,
        'predicted_L': L_pred_original,
        'L_error': L_pred_original - L_test_original,
        'L_relative_error_%': np.abs(L_pred_original - L_test_original) / (L_test_original + 1e-8) * 100,
        
        # True stress tensor
        'true_sx': stress_true_reconstructed[:, 0],
        'true_sy': stress_true_reconstructed[:, 1],
        'true_sz': stress_true_reconstructed[:, 2],
        'true_txy': stress_true_reconstructed[:, 3],
        'true_tyz': stress_true_reconstructed[:, 4],
        'true_txz': stress_true_reconstructed[:, 5],
        
        # Predicted stress tensor
        'pred_sx': stress_pred_reconstructed[:, 0],
        'pred_sy': stress_pred_reconstructed[:, 1],
        'pred_sz': stress_pred_reconstructed[:, 2],
        'pred_txy': stress_pred_reconstructed[:, 3],
        'pred_tyz': stress_pred_reconstructed[:, 4],
        'pred_txz': stress_pred_reconstructed[:, 5],
        
        # Direction vectors
        'dir_x': direction_test[:, 0],
        'dir_y': direction_test[:, 1],
        'dir_z': direction_test[:, 2],
        'dir_xy': direction_test[:, 3],
        'dir_yz': direction_test[:, 4],
        'dir_xz': direction_test[:, 5],
    })
    
    # Add stress plane information for each sample
    unified_df['stress_plane'] = unified_df['case_id'].apply(get_stress_plane_for_case)
    
    # Save unified CSV
    filename = 'unified_test_results.csv'
    filepath = os.path.join(data_dir, filename)
    unified_df.to_csv(filepath, index=False)
    
    print(f"✅ Unified test results CSV saved: {filepath}")
    print(f"   Total samples: {len(unified_df)}")
    print(f"   Case distribution: {unified_df['case_id'].value_counts().sort_index().to_dict()}")
    
    return filepath

def generate_complete_dataset_results_csv(
    L_train_true, L_train_pred, case_train_ids, direction_train, processor,
    L_val_true, L_val_pred, case_val_ids, direction_val,
    L_test_true, L_test_pred, case_test_ids, direction_test,
    output_dir
):
    """
    Generate complete dataset prediction results CSV containing train/val/test sets
    
    Args:
        L_train_true, L_train_pred: Training set true and predicted values
        case_train_ids: Training set case IDs
        direction_train: Training set direction vectors
        processor: Data processor
        L_val_true, L_val_pred: Validation set true and predicted values
        case_val_ids: Validation set case IDs
        direction_val: Validation set direction vectors
        L_test_true, L_test_pred: Test set true and predicted values
        case_test_ids: Test set case IDs
        direction_test: Test set direction vectors
        output_dir: Output directory
    
    Returns:
        str: Saved CSV file path
    """
    results_list = []
    
    # Process training set
    stress_train = processor.reconstruct_stress_tensor(L_train_pred, direction_train)
    for i in range(len(L_train_true)):
        results_list.append({
            'dataset_type': 'train',
            'case_id': case_train_ids[i],
            'L_true': L_train_true[i],
            'L_pred': L_train_pred[i],
            'L_error': L_train_pred[i] - L_train_true[i],
            'L_relative_error_%': np.abs(L_train_pred[i] - L_train_true[i]) / (L_train_true[i] + 1e-8) * 100,
            'sigma_11': stress_train[i, 0],
            'sigma_22': stress_train[i, 1],
            'sigma_33': stress_train[i, 2],
            'sigma_12': stress_train[i, 3],
            'sigma_13': stress_train[i, 4],
            'sigma_23': stress_train[i, 5],
            'dir_1': direction_train[i, 0],
            'dir_2': direction_train[i, 1],
            'dir_3': direction_train[i, 2],
            'dir_4': direction_train[i, 3],
            'dir_5': direction_train[i, 4],
            'dir_6': direction_train[i, 5]
        })
    
    # Process validation set (if exists)
    if len(L_val_true) > 0:
        stress_val = processor.reconstruct_stress_tensor(L_val_pred, direction_val)
        for i in range(len(L_val_true)):
            results_list.append({
                'dataset_type': 'val',
                'case_id': case_val_ids[i],
                'L_true': L_val_true[i],
                'L_pred': L_val_pred[i],
                'L_error': L_val_pred[i] - L_val_true[i],
                'L_relative_error_%': np.abs(L_val_pred[i] - L_val_true[i]) / (L_val_true[i] + 1e-8) * 100,
                'sigma_11': stress_val[i, 0],
                'sigma_22': stress_val[i, 1],
                'sigma_33': stress_val[i, 2],
                'sigma_12': stress_val[i, 3],
                'sigma_13': stress_val[i, 4],
                'sigma_23': stress_val[i, 5],
                'dir_1': direction_val[i, 0],
                'dir_2': direction_val[i, 1],
                'dir_3': direction_val[i, 2],
                'dir_4': direction_val[i, 3],
                'dir_5': direction_val[i, 4],
                'dir_6': direction_val[i, 5]
            })
    
    # Process test set
    stress_test = processor.reconstruct_stress_tensor(L_test_pred, direction_test)
    for i in range(len(L_test_true)):
        results_list.append({
            'dataset_type': 'test',
            'case_id': case_test_ids[i],
            'L_true': L_test_true[i],
            'L_pred': L_test_pred[i],
            'L_error': L_test_pred[i] - L_test_true[i],
            'L_relative_error_%': np.abs(L_test_pred[i] - L_test_true[i]) / (L_test_true[i] + 1e-8) * 100,
            'sigma_11': stress_test[i, 0],
            'sigma_22': stress_test[i, 1],
            'sigma_33': stress_test[i, 2],
            'sigma_12': stress_test[i, 3],
            'sigma_13': stress_test[i, 4],
            'sigma_23': stress_test[i, 5],
            'dir_1': direction_test[i, 0],
            'dir_2': direction_test[i, 1],
            'dir_3': direction_test[i, 2],
            'dir_4': direction_test[i, 3],
            'dir_5': direction_test[i, 4],
            'dir_6': direction_test[i, 5]
        })
    
    # Create DataFrame and save
    results_df = pd.DataFrame(results_list)
    
    # Save path
    csv_filename = get_timestamped_filename('complete_dataset_results', 'csv')
    csv_path = os.path.join(output_dir, 'csv_data', csv_filename)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ Complete dataset results saved: {csv_path}")
    print(f"   - Training set: {len(L_train_true)} samples")
    print(f"   - Validation set: {len(L_val_true)} samples")
    print(f"   - Test set: {len(L_test_true)} samples")
    print(f"   - Total: {len(results_df)} samples")
    
    return csv_path

def plot_train_test_comparison(
    L_train_true, L_train_pred,
    L_test_true, L_test_pred,
    save_path=None,
    title_prefix=""
):
    """
    Plot training and test set prediction comparison scatter plots
    
    Args:
        L_train_true, L_train_pred: Training set true and predicted values
        L_test_true, L_test_pred: Test set true and predicted values
        save_path: Save path
        title_prefix: Title prefix
    """
    from sklearn.metrics import r2_score, mean_squared_error
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training set scatter plot
    train_r2 = r2_score(L_train_true, L_train_pred)
    train_rmse = np.sqrt(mean_squared_error(L_train_true, L_train_pred))
    
    ax1.scatter(L_train_true, L_train_pred, alpha=0.6, s=30, c='blue', label='Training Set')
    ax1.plot([L_train_true.min(), L_train_true.max()], 
             [L_train_true.min(), L_train_true.max()], 
             'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('True L (MPa)', fontsize=12)
    ax1.set_ylabel('Predicted L (MPa)', fontsize=12)
    ax1.set_title(f'{title_prefix} - Training Set\nR²={train_r2:.4f}, RMSE={train_rmse:.2f} MPa', 
                  fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Test set scatter plot
    test_r2 = r2_score(L_test_true, L_test_pred)
    test_rmse = np.sqrt(mean_squared_error(L_test_true, L_test_pred))
    
    ax2.scatter(L_test_true, L_test_pred, alpha=0.6, s=30, c='green', label='Test Set')
    ax2.plot([L_test_true.min(), L_test_true.max()], 
             [L_test_true.min(), L_test_true.max()], 
             'r--', lw=2, label='Perfect Prediction')
    ax2.set_xlabel('True L (MPa)', fontsize=12)
    ax2.set_ylabel('Predicted L (MPa)', fontsize=12)
    ax2.set_title(f'{title_prefix} - Test Set\nR²={test_r2:.4f}, RMSE={test_rmse:.2f} MPa', 
                  fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Train/test comparison plot saved: {save_path}")
    
    plt.close()
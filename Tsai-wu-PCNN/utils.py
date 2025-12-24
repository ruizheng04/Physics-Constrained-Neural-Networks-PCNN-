# utils.py
# Responsibility: Utility functions including plotting, evaluation metrics calculation, etc.

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score

def setup_plotting():
    """Setup matplotlib plotting environment"""
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
    """Calculate failure length L values"""
    return np.sqrt(np.sum(stress_tensor**2, axis=1))

def plot_L_prediction_analysis(model, processor, X_test, y_test, case_test_ids, save_dir=None, **kwargs):
    """Simplified L value prediction analysis plotting"""
    print("📊 Simplified L value prediction analysis...")
    
    # Basic analysis and plotting logic
    # Add specific plotting code here as needed
    
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
        
        # Add colorbar to the last subplot
        if i == len(stress_names) - 1:
            plt.colorbar(scatter, ax=ax, label='Case ID')
    
    plt.tight_layout()
    
    if save_dir:
        filename = get_timestamped_filename('stress_component_analysis', 'png')
        save_path = os.path.join(save_dir, 'plots', filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Stress component analysis saved: {save_path}")
    
    plt.show()
    
    return component_metrics

def plot_training_history(history, save_path=None):
    """Plot decoupled prediction training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Decoupled Prediction Training History', fontsize=16)
    
    # Loss curves
    ax1 = axes[0, 0]
    ax1.plot(history['train_losses'], label='Training Loss', alpha=0.7)
    ax1.plot(history['val_losses'], label='Validation Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # L value loss
    ax2 = axes[0, 1]
    ax2.plot(history['L_losses'], label='L Training Loss', alpha=0.7)
    ax2.plot(history['val_L_losses'], label='L Validation Loss', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('L Loss')
    ax2.set_title('L Value Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Physics loss and Lambda
    ax3 = axes[1, 0]
    ax3.plot(history['physics_losses'], label='Physics Loss', alpha=0.7, color='red')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Physics Loss')
    ax3.set_title('Physics Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(history['lambda_values'], label='Lambda', alpha=0.7, color='orange')
    ax3_twin.set_ylabel('Lambda Value')
    ax3_twin.legend(loc='upper right')
    
    # Learning rate
    ax4 = axes[1, 1]
    ax4.plot(history['learning_rates'], alpha=0.7, color='green')
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

def comprehensive_model_evaluation(model, processor, X_test, y_test, case_test_ids, tsai_wu_coeffs, save_dir, **kwargs):
    """Simplified model evaluation function - decoupled prediction mode"""
    print("🔍 Executing simplified evaluation for decoupled prediction mode...")
    
    # In decoupled prediction mode, this function is mainly for interface compatibility
    # Actual evaluation is done directly in main.py
    
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
        # Predict normalized stress
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
    
    # Calculate L value metrics
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
    
    if output_folders and isinstance(output_folders, str):
        checkpoint_dir = os.path.join(output_folders, 'data')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save prediction results
        filename = get_timestamped_filename('checkpoint_predictions', 'csv')
        results_file = os.path.join(checkpoint_dir, filename)
        import pandas as pd
        results_df = pd.DataFrame(y_pred_original, columns=stress_names)
        results_df['L_true'] = L_true
        results_df['L_pred'] = L_pred
        results_df['case_id'] = case_test_ids
        results_df.to_csv(results_file, index=False)
        print(f"  Prediction results saved: {results_file}")
    
    return metrics

def plot_L_prediction_scatter(L_true, L_pred, case_ids=None, save_path=None, title_prefix="Decoupled Prediction"):
    """Plot L value prediction scatter plot"""
    print("📊 Plotting L value prediction scatter...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'{title_prefix} - L Value Prediction Analysis', fontsize=16, fontweight='bold')
    
    # 1. Main scatter plot - L prediction vs L true
    ax1 = axes[0]
    
    if case_ids is not None:
        # Color by case_id
        scatter = ax1.scatter(L_true, L_pred, c=case_ids, alpha=0.7, s=50, cmap='tab10', edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax1, label='Case ID')
    else:
        # Monochrome scatter plot
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
    
    # Set figure attributes
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
        print(f"L value prediction scatter saved: {save_path}")
    
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
    print("📊 Plotting L value prediction analysis by case...")
    
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
    
    # Hide extra subplots
    for i in range(n_cases, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"L value prediction by case saved: {save_path}")
    
    plt.show()
    
    return case_metrics

def plot_training_convergence(history, save_path=None):
    """Plot training convergence curves"""
    print("📈 Plotting training convergence analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Convergence Analysis', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # 1. Total loss convergence
    ax1 = axes[0, 0]
    ax1.semilogy(epochs, history['train_losses'], label='Training Loss', alpha=0.8)
    ax1.semilogy(epochs, history['val_losses'], label='Validation Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss (log scale)')
    ax1.set_title('Total Loss Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. L value loss convergence
    ax2 = axes[0, 1]
    ax2.semilogy(epochs, history['L_losses'], label='L Training Loss', alpha=0.8)
    ax2.semilogy(epochs, history['val_L_losses'], label='L Validation Loss', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('L Loss (log scale)')
    ax2.set_title('L Value Loss Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Physics loss and Lambda weight
    ax3 = axes[1, 0]
    ax3.semilogy(epochs, history['physics_losses'], label='Physics Loss', alpha=0.8, color='red')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Physics Loss (log scale)')
    ax3.set_title('Physics Loss Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add auxiliary axis for Lambda weight
    ax3_twin = ax3.twinx()
    ax3_twin.plot(epochs, history['lambda_values'], label='Lambda Weight', alpha=0.7, color='orange', linewidth=2)
    ax3_twin.set_ylabel('Lambda Weight')
    ax3_twin.legend(loc='upper right')
    
    # 4. Learning rate changes
    ax4 = axes[1, 1]
    ax4.semilogy(epochs, history['learning_rates'], alpha=0.8, color='green')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate (log scale)')
    ax4.set_title('Learning Rate Schedule')
    ax4.grid(True, alpha=0.3)
    
    # Add vertical lines for key training stages
    if 'lambda_values' in history:
        # Find the point where physics loss starts to intervene
        lambda_start = None
        for i, lambda_val in enumerate(history['lambda_values']):
            if lambda_val > 0:
                lambda_start = i
                break
        
        if lambda_start:
            for ax in [ax1, ax2, ax3, ax4]:
                ax.axvline(lambda_start, color='purple', linestyle='--', alpha=0.5, label='Physics Loss Start')
                if ax == ax1:  # Only show legend on the first plot
                    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training convergence analysis saved: {save_path}")
    
    plt.show()

def plot_dual_physics_analysis(model, processor, X_test, y_test, case_test_ids, direction_test, save_dir=None):
    """Comprehensive analysis plotting for dual physics constraints"""
    print("📊 Plotting dual physics constraint comprehensive analysis...")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Prediction
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        L_pred_scaled = model(X_test_tensor).cpu().numpy().flatten()
    
    # Transform to original scale
    L_pred_original = processor.inverse_transform_L(L_pred_scaled)
    L_test_original = processor.inverse_transform_L(y_test)
    
    # Reconstruct stress tensor
    stress_reconstructed = processor.reconstruct_stress_tensor(L_pred_original, direction_test)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Dual Physics Constrained PINN Analysis', fontsize=16, fontweight='bold')
    
    # 1. L value prediction scatter plot
    ax1 = axes[0, 0]
    scatter = ax1.scatter(L_test_original, L_pred_original, c=case_test_ids, alpha=0.7, s=30, cmap='tab10')
    min_L, max_L = min(L_test_original.min(), L_pred_original.min()), max(L_test_original.max(), L_pred_original.max())
    ax1.plot([min_L, max_L], [min_L, max_L], 'r--', alpha=0.8, linewidth=2)
    
    from sklearn.metrics import r2_score
    r2 = r2_score(L_test_original, L_pred_original)
    rmse = np.sqrt(np.mean((L_test_original - L_pred_original)**2))
    
    ax1.set_xlabel('True L (MPa)')
    ax1.set_ylabel('Predicted L (MPa)')
    ax1.set_title(f'L Value Prediction\nR² = {r2:.4f}, RMSE = {rmse:.2f}')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Case ID')
    
    # 2. Relative error distribution
    ax2 = axes[0, 1]
    relative_error = np.abs(L_pred_original - L_test_original) / (L_test_original + 1e-8) * 100
    ax2.hist(relative_error, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(relative_error.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {relative_error.mean():.1f}%')
    ax2.set_xlabel('Relative Error (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Relative Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Stress component reconstruction comparison
    ax3 = axes[0, 2]
    stress_names = ['σx', 'σy', 'σz', 'τxy', 'τyz', 'τxz']
    original_stress = L_test_original.reshape(-1, 1) * direction_test  # True stress reconstruction
    
    # Select main stress component for comparison
    stress_component_idx = 0  # σx component
    ax3.scatter(original_stress[:, stress_component_idx], stress_reconstructed[:, stress_component_idx], 
               alpha=0.6, s=20)
    min_stress = min(original_stress[:, stress_component_idx].min(), stress_reconstructed[:, stress_component_idx].min())
    max_stress = max(original_stress[:, stress_component_idx].max(), stress_reconstructed[:, stress_component_idx].max())
    ax3.plot([min_stress, max_stress], [min_stress, max_stress], 'r--', alpha=0.8)
    ax3.set_xlabel(f'True {stress_names[stress_component_idx]} (MPa)')
    ax3.set_ylabel(f'Reconstructed {stress_names[stress_component_idx]} (MPa)')
    ax3.set_title(f'{stress_names[stress_component_idx]} Component Reconstruction')
    ax3.grid(True, alpha=0.3)
    
    # 4. Case-wise analysis
    ax4 = axes[1, 0]
    unique_cases = np.unique(case_test_ids)
    case_errors = []
    case_labels = []
    
    for case in unique_cases:
        case_mask = case_test_ids == case
        case_error = relative_error[case_mask]
        case_errors.append(case_error)
        case_labels.append(f'Case {case}')
    
    ax4.boxplot(case_errors, labels=case_labels)
    ax4.set_xlabel('Case')
    ax4.set_ylabel('Relative Error (%)')
    ax4.set_title('Error Distribution by Case')
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.get_xticklabels(), rotation=45)
    
    # 5. Stress magnitude comparison
    ax5 = axes[1, 1]
    L_reconstructed = np.sqrt(np.sum(stress_reconstructed**2, axis=1))
    ax5.scatter(L_test_original, L_reconstructed, alpha=0.6, s=20)
    ax5.plot([L_test_original.min(), L_test_original.max()], 
             [L_test_original.min(), L_test_original.max()], 'r--', alpha=0.8)
    ax5.set_xlabel('True L (MPa)')
    ax5.set_ylabel('Reconstructed L (MPa)')
    ax5.set_title('Stress Magnitude Consistency')
    ax5.grid(True, alpha=0.3)
    
    # 6. Prediction error vs stress magnitude
    ax6 = axes[1, 2]
    ax6.scatter(L_test_original, relative_error, alpha=0.6, s=20)
    ax6.set_xlabel('True L (MPa)')
    ax6.set_ylabel('Relative Error (%)')
    ax6.set_title('Error vs Stress Magnitude')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        filename = get_timestamped_filename('dual_physics_comprehensive_analysis', 'png')
        save_path = os.path.join(save_dir, 'plots', filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dual physics constraint analysis saved: {save_path}")
    
    plt.show()
    
    # Return detailed evaluation metrics
    return {
        'L_r2': r2,
        'L_rmse': rmse,
        'mean_relative_error': relative_error.mean(),
        'std_relative_error': relative_error.std(),
        'median_relative_error': np.median(relative_error),
        'reconstruction_consistency': r2_score(L_test_original, L_reconstructed)
    }

def evaluate_dual_physics_constraints(stress_tensor, case_ids, tsai_wu_coeffs, cuntze_params):
    """Evaluate violation degree of dual physics constraints"""
    print("🔍 Evaluating dual physics constraint violations...")
    
    tsai_wu_violations = []
    cuntze_violations = []
    
    for i, case_id in enumerate(case_ids):
        case_id = int(case_id)
        if case_id not in tsai_wu_coeffs:
            case_id = 1
        
        stress = stress_tensor[i]
        
        # Tsai-Wu criterion evaluation
        coeffs = tsai_wu_coeffs[case_id]
        f_value = (coeffs['F1'] * stress[0] + coeffs['F2'] * stress[1] + coeffs['F3'] * stress[2] +
                  coeffs['F11'] * stress[0]**2 + coeffs['F22'] * stress[1]**2 + coeffs['F33'] * stress[2]**2 +
                  coeffs['F44'] * stress[3]**2 + coeffs['F55'] * stress[4]**2 + coeffs['F66'] * stress[5]**2 +
                  2 * coeffs['F12'] * stress[0] * stress[1] +
                  2 * coeffs['F13'] * stress[0] * stress[2] +
                  2 * coeffs['F23'] * stress[1] * stress[2])
        
        tsai_wu_violation = max(0, f_value - 1.0)
        tsai_wu_violations.append(tsai_wu_violation)
        
        # Cuntze criterion evaluation
        if case_id in cuntze_params:
            params = cuntze_params[case_id]
            
            # Fiber failure
            fiber_failure = 0
            if stress[0] > 0:
                fiber_failure += max(0, stress[0] / params['S_f_parallel_t'] - 1.0)
            else:
                fiber_failure += max(0, abs(stress[0]) / params['S_f_parallel_c'] - 1.0)
            
            # Matrix failure
            matrix_stress = np.sqrt(stress[1]**2 + stress[2]**2 + stress[3]**2 + stress[4]**2 + stress[5]**2)
            matrix_failure = max(0, matrix_stress / params['S_f_perpendicular'] - 1.0)
            
            cuntze_violation = fiber_failure + matrix_failure
        else:
            cuntze_violation = 0
        
        cuntze_violations.append(cuntze_violation)
    
    tsai_wu_violations = np.array(tsai_wu_violations)
    cuntze_violations = np.array(cuntze_violations)
    
    print(f"   Tsai-Wu violation samples: {(tsai_wu_violations > 0).sum()}/{len(tsai_wu_violations)}")
    print(f"   Tsai-Wu average violation: {tsai_wu_violations.mean():.4f}")
    print(f"   Cuntze violation samples: {(cuntze_violations > 0).sum()}/{len(cuntze_violations)}")
    print(f"   Cuntze average violation: {cuntze_violations.mean():.4f}")
    
    return {
        'tsai_wu_violations': tsai_wu_violations,
        'cuntze_violations': cuntze_violations,
        'tsai_wu_violation_rate': (tsai_wu_violations > 0).mean(),
        'cuntze_violation_rate': (cuntze_violations > 0).mean(),
        'mean_tsai_wu_violation': tsai_wu_violations.mean(),
        'mean_cuntze_violation': cuntze_violations.mean()
    }
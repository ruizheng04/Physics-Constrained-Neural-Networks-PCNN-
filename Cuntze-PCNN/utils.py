# utils.py
# Responsibility: Store auxiliary functions such as plotting and evaluation metrics.

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
    # Can add specific plotting code as needed
    
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
    
    # Convert to original scale
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
    """Plot training history with SA-PINN weight visualization support"""
    
    # Check if SA-PINN weight data is included
    has_sa_weights = 'sa_pinn_weights' in history
    
    if has_sa_weights:
        # SA-PINN mode: 5 subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
    else:
        # Standard mode: 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
    
    # Fix key name issues - use correct key names
    if 'train_losses' in history:
        epochs = range(1, len(history['train_losses']) + 1)
        train_loss_key = 'train_losses'
    elif 'train_loss' in history:
        epochs = range(1, len(history['train_loss']) + 1)
        train_loss_key = 'train_loss'
    else:
        print("⚠️ Loss data not found in training history")
        return
    
    # 1. Overall loss
    ax1 = axes[0]
    ax1.plot(epochs, history[train_loss_key], 'b-', alpha=0.8, label='Training Loss')
    
    # Check validation loss key name
    val_loss_key = None
    if 'val_losses' in history and len(history['val_losses']) > 0:
        val_loss_key = 'val_losses'
    elif 'val_loss' in history and len(history['val_loss']) > 0:
        val_loss_key = 'val_loss'
    
    if val_loss_key:
        ax1.plot(epochs, history[val_loss_key], 'r-', alpha=0.8, label='Validation Loss')
    
    ax1.set_xlabel('Training Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Data fitting loss
    ax2 = axes[1]
    ax2.plot(epochs, history['train_data_loss'], 'g-', alpha=0.8, label='Training Data Loss')
    if history['val_data_loss'] and len(history['val_data_loss']) > 0:
        ax2.plot(epochs, history['val_data_loss'], 'orange', alpha=0.8, label='Validation Data Loss')
    ax2.set_xlabel('Training Epoch')
    ax2.set_ylabel('Data Fitting Loss')
    ax2.set_title('Data Fitting Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Tsai-Wu physical loss
    ax3 = axes[2]
    ax3.plot(epochs, history['train_tsai_wu_loss'], 'purple', alpha=0.8, label='Tsai-Wu Loss')
    ax3.set_xlabel('Training Epoch')
    ax3.set_ylabel('Tsai-Wu Loss')
    ax3.set_title('Tsai-Wu Physical Constraint Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Cuntze physical loss
    ax4 = axes[3]
    ax4.plot(epochs, history['train_cuntze_loss'], 'brown', alpha=0.8, label='Cuntze Loss')
    ax4.set_xlabel('Training Epoch')
    ax4.set_ylabel('Cuntze Loss')
    ax4.set_title('Cuntze Physical Constraint Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # 5. SA-PINN weight evolution (only in SA-PINN mode)
    if has_sa_weights:
        ax5 = axes[4]
        sa_weights = history['sa_pinn_weights']
        weight_epochs = range(1, len(sa_weights['tsai_wu_mean']) + 1)
        
        # Tsai-Wu weight
        ax5.plot(weight_epochs, sa_weights['tsai_wu_mean'], 'b-', alpha=0.8, 
                label='Tsai-Wu Average Weight', linewidth=2)
        ax5.plot(weight_epochs, sa_weights['tsai_wu_max'], 'b--', alpha=0.6, 
                label='Tsai-Wu Maximum Weight')
        
        # Cuntze weight
        ax5.plot(weight_epochs, sa_weights['cuntze_mean'], 'r-', alpha=0.8, 
                label='Cuntze Average Weight', linewidth=2)
        ax5.plot(weight_epochs, sa_weights['cuntze_max'], 'r--', alpha=0.6, 
                label='Cuntze Maximum Weight')
        
        ax5.set_xlabel('Training Epoch')
        ax5.set_ylabel('Adaptive Weight Value')
        ax5.set_title('SA-PINN Adaptive Weight Evolution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6th subplot: weight distribution standard deviation
        if len(axes) > 5:
            ax6 = axes[5]
            ax6.plot(weight_epochs, sa_weights['tsai_wu_std'], 'b-', alpha=0.8, 
                    label='Tsai-Wu Weight Std')
            ax6.plot(weight_epochs, sa_weights['cuntze_std'], 'r-', alpha=0.8, 
                    label='Cuntze Weight Std')
            ax6.set_xlabel('Training Epoch')
            ax6.set_ylabel('Weight Standard Deviation')
            ax6.set_title('SA-PINN Weight Distribution Dispersion')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def save_model_and_results(model, history, processor, save_dir):
    """保存模型和结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unified_dir = f"{save_dir}_{timestamp}"
    
    # 创建目录结构
    os.makedirs(os.path.join(unified_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(unified_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(unified_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(unified_dir, 'reports'), exist_ok=True)
    
    # 保存模型
    model_filename = f"decoupled_pinn_model_{timestamp}.pth"
    model_path = os.path.join(unified_dir, 'models', model_filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'layer_sizes': model.layer_sizes,
            'activation': model.activation
        }
    }, model_path)
    
    # 保存训练历史
    history_filename = f"training_history_{timestamp}.pkl"
    history_path = os.path.join(unified_dir, 'data', history_filename)
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    print(f"模型已保存: {model_path}")
    print(f"训练历史已保存: {history_path}")
    
    return model_path, history_path, unified_dir

def comprehensive_model_evaluation(model, processor, X_test, y_test, case_test_ids, tsai_wu_coeffs, save_dir, **kwargs):
    """简化的模型评估函数 - 解耦预测模式"""
    print("🔍 执行解耦预测模式的简化评估...")
    
    # 在解耦预测模式下，这个函数主要用于保持接口兼容性
    # 实际的评估在main.py中直接进行
    
    return {
        'L_metrics': {'L_r2': 0.0, 'L_rmse': 0.0},
        'physics_metrics': {'constraint_r2': 0.0},
        'save_dir': save_dir
    }

def evaluate_model_at_checkpoint(model, processor, X_test, y_test, case_test_ids, output_folders):
    """在检查点评估模型性能"""
    model.eval()
    device = next(model.parameters()).device
    
    # 转换测试数据
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        # 预测标准化的应力
        y_pred_scaled = model(X_test_tensor).cpu().numpy()
    
    # 转换回原始尺度
    if processor.use_stress_normalization:
        y_pred_original = processor.inverse_transform_stress(y_pred_scaled)
        y_test_original = processor.inverse_transform_stress(y_test)
    else:
        y_pred_original = y_pred_scaled
        y_test_original = y_test
    
    # 计算评估指标
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred_original)
    
    # 计算L值指标
    L_true = calculate_L_values(y_test_original)
    L_pred = calculate_L_values(y_pred_original)
    L_r2 = r2_score(L_true, L_pred)
    L_rmse = np.sqrt(mean_squared_error(L_true, L_pred))
    
    # 按应力分量计算指标
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
    
    print(f"检查点评估结果:")
    print(f"  整体 RMSE: {rmse:.4f}")
    print(f"  整体 R²: {r2:.4f}")
    print(f"  L值 R²: {L_r2:.4f}")
    print(f"  L值 RMSE: {L_rmse:.4f}")
    
    # 如果提供了输出文件夹，保存结果
    if output_folders and isinstance(output_folders, str):
        checkpoint_dir = os.path.join(output_folders, 'data')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存预测结果
        filename = get_timestamped_filename('checkpoint_predictions', 'csv')
        results_file = os.path.join(checkpoint_dir, filename)
        import pandas as pd
        results_df = pd.DataFrame(y_pred_original, columns=stress_names)
        results_df['L_true'] = L_true
        results_df['L_pred'] = L_pred
        results_df['case_id'] = case_test_ids
        results_df.to_csv(results_file, index=False)
        print(f"  预测结果已保存: {results_file}")
    
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
    
    # Set figure properties
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
    """按case分别绘制L值预测性能"""
    print("📊 绘制按case分组的L值预测分析...")
    
    unique_cases = np.unique(case_ids)
    n_cases = len(unique_cases)
    
    # 计算子图布局
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
        
        # 散点图
        ax.scatter(L_true_case, L_pred_case, alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
        
        # 完美预测线
        min_L, max_L = min(L_true_case.min(), L_pred_case.min()), max(L_true_case.max(), L_pred_case.max())
        ax.plot([min_L, max_L], [min_L, max_L], 'r--', alpha=0.8, linewidth=2)
        
        # 计算case专用指标
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
    
    # 隐藏多余的子图
    for i in range(n_cases, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"按case分组的L值预测图已保存: {save_path}")
    
    plt.show()
    
    return case_metrics

def plot_training_convergence(history, save_path=None):
    """绘制训练收敛曲线"""
    print("📈 绘制训练收敛分析...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Convergence Analysis', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # 1. 总损失收敛
    ax1 = axes[0, 0]
    ax1.semilogy(epochs, history['train_losses'], label='Training Loss', alpha=0.8)
    if 'val_losses' in history and len(history['val_losses']) > 0:
        ax1.semilogy(epochs, history['val_losses'], label='Validation Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss (log scale)')
    ax1.set_title('Total Loss Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 数据损失收敛
    ax2 = axes[0, 1]
    ax2.semilogy(epochs, history['train_data_loss'], label='Data Training Loss', alpha=0.8)
    if 'val_data_loss' in history and len(history['val_data_loss']) > 0:
        ax2.semilogy(epochs, history['val_data_loss'], label='Data Validation Loss', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Data Loss (log scale)')
    ax2.set_title('Data Loss Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 物理损失和Lambda权重
    ax3 = axes[1, 0]
    if 'train_tsai_wu_loss' in history:
        ax3.semilogy(epochs, history['train_tsai_wu_loss'], label='Tsai-Wu Loss', alpha=0.8, color='blue')
    if 'train_cuntze_loss' in history:
        ax3.semilogy(epochs, history['train_cuntze_loss'], label='Cuntze Loss', alpha=0.8, color='red')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Physics Loss (log scale)')
    ax3.set_title('Physics Loss Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 添加Lambda权重的辅助轴
    if 'lambda_values' in history:
        ax3_twin = ax3.twinx()
        ax3_twin.plot(epochs, history['lambda_values'], label='Lambda Weight', alpha=0.7, color='orange', linewidth=2)
        ax3_twin.set_ylabel('Lambda Weight')
        ax3_twin.legend(loc='upper right')
    
    # 4. 学习率变化
    ax4 = axes[1, 1]
    if 'learning_rates' in history:
        ax4.semilogy(epochs, history['learning_rates'], alpha=0.7, color='green')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate (log scale)')
        ax4.set_title('Learning Rate Schedule')
        ax4.grid(True, alpha=0.3)
    
    # 添加关键训练阶段的标记线
    if 'lambda_values' in history:
        # 找到物理损失开始介入的时间点
        lambda_start = None
        for i, lambda_val in enumerate(history['lambda_values']):
            if lambda_val > 0:
                lambda_start = i
                break
        
        if lambda_start:
            for ax in [ax1, ax2, ax3, ax4]:
                ax.axvline(lambda_start, color='purple', linestyle='--', alpha=0.5, label='Physics Loss Start')
                if ax == ax1:  # 只在第一个图上显示图例
                    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练收敛分析图已保存: {save_path}")
    
    plt.show()

def plot_sa_pinn_weight_analysis(history, save_path=None):
    """绘制SA-PINN权重详细分析图"""
    
    if 'sa_pinn_weights' not in history:
        print("⚠️ 历史记录中未找到SA-PINN权重数据")
        return
    
    sa_weights = history['sa_pinn_weights']
    epochs = range(1, len(sa_weights['tsai_wu_mean']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 权重均值对比
    ax1 = axes[0, 0]
    ax1.plot(epochs, sa_weights['tsai_wu_mean'], 'b-', linewidth=2, label='Tsai-Wu')
    ax1.plot(epochs, sa_weights['cuntze_mean'], 'r-', linewidth=2, label='Cuntze')
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('平均权重值')
    ax1.set_title('自适应权重均值演化')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 最大权重对比
    ax2 = axes[0, 1]
    ax2.plot(epochs, sa_weights['tsai_wu_max'], 'b-', linewidth=2, label='Tsai-Wu')
    ax2.plot(epochs, sa_weights['cuntze_max'], 'r-', linewidth=2, label='Cuntze')
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('最大权重值')
    ax2.set_title('自适应权重最大值演化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 权重标准差（分布离散度）
    ax3 = axes[1, 0]
    ax3.plot(epochs, sa_weights['tsai_wu_std'], 'b-', linewidth=2, label='Tsai-Wu')
    ax3.plot(epochs, sa_weights['cuntze_std'], 'r-', linewidth=2, label='Cuntze')
    ax3.set_xlabel('训练轮次')
    ax3.set_ylabel('权重标准差')
    ax3.set_title('权重分布离散度')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 权重集中度指标（最大值/均值比）
    ax4 = axes[1, 1]
    tw_concentration = [max_w / (mean_w + 1e-8) for max_w, mean_w in 
                       zip(sa_weights['tsai_wu_max'], sa_weights['tsai_wu_mean'])]
    cuntze_concentration = [max_w / (mean_w + 1e-8) for max_w, mean_w in 
                           zip(sa_weights['cuntze_max'], sa_weights['cuntze_mean'])]
    
    ax4.plot(epochs, tw_concentration, 'b-', linewidth=2, label='Tsai-Wu')
    ax4.plot(epochs, cuntze_concentration, 'r-', linewidth=2, label='Cuntze')
    ax4.set_xlabel('训练轮次')
    ax4.set_ylabel('权重集中度 (最大值/均值)')
    ax4.set_title('权重集中度指标')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ SA-PINN权重分析图保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()
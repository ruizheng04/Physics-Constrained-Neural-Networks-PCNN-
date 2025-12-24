# main.py
# Responsibility: Dual physical constraint PINN main program - Tsai-Wu + Cuntze criteria

import sys
import warnings
import pandas as pd
import os
import torch
import numpy as np

# Import configuration and modules
from config import FIXED_CONFIG, TEST_CONFIG, OPTIMAL_CONFIG
from data_processor import TsaiWuPINNDataProcessor
from dual_trainer import train_dual_physics_pinn

def setup_plotting():
    """Setup matplotlib plotting environment"""
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.family'] = 'sans-serif'

def main():
    """Main function - Dual physical constraint PINN (Tsai-Wu + Cuntze)"""
    print("\n" + "="*100)
    print("🚀 Dual Physics Constrained PINN (Tsai-Wu + Cuntze Criteria)")
    print("📋 Input: 20D Enhanced Features (14 Material + 6 Direction)")
    print("📊 Output: 1D Failure Length L")
    print("🔧 Core Strategy: Predict L → Reconstruct Stress → Dual Physics Validation")
    print("⚖️  Physics Constraints: Tsai-Wu Criterion + Cuntze Criterion")
    print("🎯 Advantage: Comprehensive Material Failure Prediction")
    print("🏆 Features: SA-PINN Adaptive Weighting + Progressive Constraint Activation")
    print("="*100)
    
    setup_plotting()
    warnings.filterwarnings('ignore')
    
    # Mode selection
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("\n🧪 Starting dual physical constraint test mode...")
        config = TEST_CONFIG
        save_dir = 'results_dual_test'
    elif len(sys.argv) > 1 and sys.argv[1] == '--optimal':
        print("\n🏆 Starting dual physical constraint optimal mode...")
        config = OPTIMAL_CONFIG
        save_dir = 'results_dual_optimal'
    elif len(sys.argv) > 1 and sys.argv[1] == '--sa':
        print("\n🤖 Starting SA-PINN adaptive weight mode...")
        config = OPTIMAL_CONFIG.copy()
        config['use_sa_pinn'] = True
        save_dir = 'results_dual_sa_pinn'
        print(f"✅ SA-PINN adaptive weighting enabled")
    else:
        print("\n⚙️ Starting dual physical constraint standard mode...")
        config = FIXED_CONFIG
        save_dir = 'results_dual_standard'
    
    print(f"\n🔧 Dual physical constraint configuration:")
    print(f"   Tsai-Wu weight: λ={config.get('lambda_tsai_wu_final', 'N/A')}, W={config.get('W_tsai_wu', 'N/A')}")
    print(f"   Cuntze weight: λ={config.get('lambda_cuntze_final', 'N/A')}, W={config.get('W_cuntze', 'N/A')}")
    print(f"   SA-PINN: {'Enabled' if config.get('use_sa_pinn', False) else 'Disabled'}")
    print(f"   Network architecture: {config.get('layer_sizes', 'N/A')}")
    
    # 1. Data processing
    processor = TsaiWuPINNDataProcessor()
    try:
        df = processor.load_data('datasetnew.csv')
    except FileNotFoundError:
        print("❌ Error: 'datasetnew.csv' not found. Please ensure data file is in current directory.")
        return
    
    # Prepare decoupled prediction data
    enhanced_features, L_values, case_ids, direction_vectors, original_stress = processor.prepare_decoupled_features_and_labels(df)
    
    # Dataset split
    print(f"\n📊 Dual physical constraint dataset split...")
    split_result = processor.split_by_case_and_merge(
        enhanced_features, L_values, case_ids, test_size=0.2, val_size=0.1
    )
    
    # Unpack split results
    if len(split_result) == 13:
        (X_train_df, X_val_df, X_test_df, 
         L_train_df, L_val_df, L_test_df,
         case_train_ids, case_val_ids, case_test_ids, case_info,
         direction_train, direction_val, direction_test) = split_result
    else:
        # Backward compatibility
        (X_train_df, X_val_df, X_test_df, 
         L_train_df, L_val_df, L_test_df,
         case_train_ids, case_val_ids, case_test_ids, case_info) = split_result
        
        direction_df = pd.DataFrame(direction_vectors, columns=[f'dir_{i}' for i in range(6)])
        train_indices = X_train_df.index.tolist()
        val_indices = X_val_df.index.tolist() if len(X_val_df) > 0 else []
        test_indices = X_test_df.index.tolist()
        
        direction_train = direction_df.iloc[train_indices].values
        direction_val = direction_df.iloc[val_indices].values if len(val_indices) > 0 else np.array([]).reshape(0, 6)
        direction_test = direction_df.iloc[test_indices].values
    
    # Data standardization
    X_train, L_train = processor.fit_transform(X_train_df, L_train_df)
    X_val, L_val = processor.transform(X_val_df, L_val_df) if len(X_val_df) > 0 else (np.array([]).reshape(0, 20), np.array([]))
    X_test, L_test = processor.transform(X_test_df, L_test_df)
    
    print(f"\n✅ Dual physical constraint data preparation complete:")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Validation set: {X_val.shape[0]} samples") 
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # 2. Dual physical constraint model training
    print(f"\n🎯 Starting dual physical constraint training, results saved to: {save_dir}")
    
    model, history = train_dual_physics_pinn(
        X_train, L_train, case_train_ids, direction_train,
        X_val, L_val, case_val_ids, direction_val,
        processor, config,
        X_test=X_test, L_test=L_test,
        case_test_ids=case_test_ids, direction_test=direction_test,
        output_folders=save_dir
    )
    
    # 3. Save model and results
    from utils import save_model_and_results, get_timestamped_filename
    print(f"\n💾 Saving dual physical constraint model...")
    model_path, history_path, unified_dir = save_model_and_results(model, history, processor, save_dir)
    
    # 4. Plot training history
    print(f"\n📈 Plotting dual physical constraint training history...")
    plot_dual_training_history(history, unified_dir)
    
    # 5. Model evaluation
    print(f"\n🔍 Starting dual physical constraint model evaluation...")
    evaluate_dual_physics_model(model, processor, X_test, L_test, direction_test, case_test_ids, unified_dir)
    
    print(f"\n🎉 Dual physical constraint PINN training and evaluation complete!")
    print(f"📁 All results saved to: {unified_dir}")

def plot_dual_training_history(history, save_dir):
    """Plot dual physical constraint training history"""
    import matplotlib.pyplot as plt
    from utils import get_timestamped_filename
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Dual Physics Constrained PINN Training History', fontsize=16)
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Total loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_losses'], label='Training', alpha=0.8)
    ax1.plot(epochs, history['val_losses'], label='Validation', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # L value loss
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['L_losses'], label='L Training', alpha=0.8)
    ax2.plot(epochs, history['val_L_losses'], label='L Validation', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('L Loss')
    ax2.set_title('L Value Loss Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Physics loss
    ax3 = axes[0, 2]
    ax3.plot(epochs, history['tsai_wu_losses'], label='Tsai-Wu', alpha=0.8, color='red')
    ax3.plot(epochs, history['cuntze_losses'], label='Cuntze', alpha=0.8, color='orange')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Physics Loss')
    ax3.set_title('Physics Constraints Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Lambda weight
    ax4 = axes[1, 0]
    ax4.plot(epochs, history['lambda_tsai_wu_values'], label='λ Tsai-Wu', alpha=0.8, color='blue')
    ax4.plot(epochs, history['lambda_cuntze_values'], label='λ Cuntze', alpha=0.8, color='green')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Lambda Weight')
    ax4.set_title('Physics Weight Schedule')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # SA-PINN weight
    ax5 = axes[1, 1]
    if 'sa_tsai_wu_weights' in history:
        ax5.plot(epochs, history['sa_tsai_wu_weights'], label='SA Tsai-Wu', alpha=0.8, color='purple')
        ax5.plot(epochs, history['sa_cuntze_weights'], label='SA Cuntze', alpha=0.8, color='brown')
        ax5.set_title('SA-PINN Adaptive Weights')
    else:
        ax5.text(0.5, 0.5, 'SA-PINN Disabled', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('SA-PINN Weights (Disabled)')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('SA Weight')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Learning rate
    ax6 = axes[1, 2]
    ax6.plot(epochs, history['learning_rates'], alpha=0.8, color='black')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Learning Rate')
    ax6.set_title('Learning Rate Schedule')
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')
    
    plt.tight_layout()
    
    # Save image
    filename = get_timestamped_filename('dual_physics_training_history', 'png')
    save_path = os.path.join(save_dir, 'plots', filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Dual physics constraint training history plot saved: {save_path}")
    plt.show()

def evaluate_dual_physics_model(model, processor, X_test, L_test, direction_test, case_test_ids, save_dir):
    """Evaluate dual physical constraint model"""
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Model prediction
    model.eval()
    device = next(model.parameters()).device
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        L_pred_norm = model(X_test_tensor).cpu().numpy().flatten()
    
    # Convert back to original scale
    L_pred_original = processor.inverse_transform_L(L_pred_norm)
    L_test_original = processor.inverse_transform_L(L_test)
    
    # Calculate evaluation metrics
    L_mse = mean_squared_error(L_test_original, L_pred_original)
    L_rmse = np.sqrt(L_mse)
    L_r2 = r2_score(L_test_original, L_pred_original)
    relative_error = np.abs(L_pred_original - L_test_original) / (L_test_original + 1e-8) * 100
    
    print(f"\n📊 Dual physical constraint model performance:")
    print(f"   L value prediction R² = {L_r2:.4f}")
    print(f"   L value prediction RMSE = {L_rmse:.4f} MPa")
    print(f"   Mean relative error = {relative_error.mean():.2f}% ± {relative_error.std():.2f}%")
    
    # Reconstruct stress tensor
    stress_reconstructed = processor.reconstruct_stress_tensor(L_pred_original, direction_test)
    print(f"   Reconstructed stress range: [{stress_reconstructed.min():.2f}, {stress_reconstructed.max():.2f}] MPa")
    
    # Plot prediction scatter plot
    from utils import plot_L_prediction_scatter
    from utils import get_timestamped_filename
    
    scatter_plot_filename = get_timestamped_filename('dual_physics_L_prediction_scatter', 'png')
    scatter_plot_path = os.path.join(save_dir, 'plots', scatter_plot_filename)
    
    plot_L_prediction_scatter(
        L_test_original, L_pred_original,
        case_ids=case_test_ids.values if hasattr(case_test_ids, 'values') else case_test_ids,
        save_path=scatter_plot_path,
        title_prefix="Dual Physics PINN"
    )

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("Usage:")
        print("  python main.py              # Standard dual physical constraint mode")
        print("  python main.py --test       # Test mode")
        print("  python main.py --optimal    # Optimal configuration mode")
        print("  python main.py --sa         # SA-PINN adaptive weight mode")
    else:
        main()
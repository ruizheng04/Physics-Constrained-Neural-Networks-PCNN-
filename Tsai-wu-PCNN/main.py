# main.py
# Responsibility: Program entry point, coordinates all modules, executes complete workflow.

import sys
import warnings
import pandas as pd
import os
import torch
import numpy as np

# Import from modules
from config import FIXED_CONFIG, TEST_CONFIG, TSAI_WU_COEFFICIENTS
from data_processor import TsaiWuPINNDataProcessor
from trainer import train_decoupled_tsai_wu_pinn
from utils import setup_plotting, plot_training_history, save_model_and_results, get_timestamped_filename

# Optimal configuration based on search results
OPTIMAL_CONFIG = {
    # Network architecture - based on search result rank 1
    'layer_sizes': [20, 512, 1],
    'activation': 'swiglu',
    'dropout_rate': 0.01,
    'use_batch_norm': False,
    'use_residual': False,
    
    # Optimizer settings - based on search results
    'lr': 0.01,
    'weight_decay': 0.0,
    'optimizer': 'AdamW',
    
    # Training settings
    'max_epochs': 3000,
    'early_stop_patience': 500,
    'warm_start_epochs': 800,
    'physics_delay_epochs': 1500,
    
    # Physical constraint weights - based on search result rank 1
    'lambda_final': 0.5,
    'W_physics': 0.2,
    'adaptive_balance': False,
    'gradient_balance_alpha': 0.1,
    'physics_warmup_rate': 0.01,
    
    # Data processing
    'normalization_method': 'minmax_columnwise',
    'batch_size': 64,
    
    # Regularization
    'l1_lambda': 0.0,
    'l2_lambda': 0.0
}

def main():
    """Main function - Decoupled prediction Tsai-Wu criterion constrained PINN"""
    print("\n" + "="*90)
    print("🚀 Decoupled Prediction Tsai-Wu PINN Model (Optimized)")
    print("📋 Input: 20D Enhanced Features (14 Material + 6 Direction)")
    print("📊 Output: 1D Failure Length L")
    print("🔧 Core Strategy: Predict L → Reconstruct Stress → Validate Physics")
    print("⚖️  Physics Constraints: Tsai-Wu Criterion on Reconstructed Stress")
    print("🎯 Advantage: Dramatically Simplified Learning Task")
    print("🏆 Configuration: Optimal Hyperparameters (R²≈0.91, RMSE≈135 MPa)")
    print("="*90)
    
    setup_plotting()
    warnings.filterwarnings('ignore')
    
    # Select mode based on command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--search':
        print("\n🔍 Starting hyperparameter search mode...")
        from hyperparameter_search import run_hyperparameter_search
        run_hyperparameter_search()
        return
    elif len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("\n🧪 Starting decoupled prediction test mode...")
        config = TEST_CONFIG
        save_dir = 'results_test_decoupled'
    elif len(sys.argv) > 1 and sys.argv[1] == '--optimal':
        print("\n🏆 Starting optimal configuration mode...")
        config = OPTIMAL_CONFIG
        save_dir = 'results_optimal_decoupled'
        print(f"🔧 Using optimal hyperparameters from search:")
        print(f"   Architecture: {config['layer_sizes']} (wide-shallow)")
        print(f"   Activation: {config['activation']}")
        print(f"   Learning rate: {config['lr']}")
        print(f"   Dropout rate: {config['dropout_rate']}")
        print(f"   Physics loss weight: λ={config['lambda_final']}, W={config['W_physics']}")
        print(f"   Expected performance: R²≈0.91, RMSE≈135 MPa")
    else:
        print("\n⚙️ Starting decoupled prediction standard mode...")
        config = FIXED_CONFIG
        save_dir = 'results_decoupled'

    print(f"🔧 Physics weight setting: W_physics = {config.get('W_physics', 0.0)}")
    print(f"🧠 Activation function: {config.get('activation', 'swiglu')}")
    print("✅ Decoupled prediction: 20D features → 1D L value → 6D stress reconstruction")

    # 1. Decoupled prediction data processing
    processor = TsaiWuPINNDataProcessor()
    try:
        df = processor.load_data('datasetnew.csv')
    except FileNotFoundError:
        print("❌ Error: 'datasetnew.csv' not found. Please ensure data file is in current directory.")
        return
        
    # Use decoupled prediction data preparation method
    enhanced_features, L_values, case_ids, direction_vectors, original_stress = processor.prepare_decoupled_features_and_labels(df)
    
    # Split dataset (need to handle features, L values, direction vectors simultaneously)
    print(f"\n📊 Decoupled prediction dataset split...")
    split_result = processor.split_by_case_and_merge(
        enhanced_features, L_values, case_ids, test_size=0.2, val_size=0.1
    )
    
    # Unpack all split results
    if len(split_result) == 13:  # New version includes direction vectors
        (X_train_df, X_val_df, X_test_df, 
         L_train_df, L_val_df, L_test_df,
         case_train_ids, case_val_ids, case_test_ids, case_info,
         direction_train, direction_val, direction_test) = split_result
        print("✅ Using unified split method, direction vectors synchronized")
    else:  # Backward compatibility with old version
        (X_train_df, X_val_df, X_test_df, 
         L_train_df, L_val_df, L_test_df,
         case_train_ids, case_val_ids, case_test_ids, case_info) = split_result
        
        # Manually split direction vectors (compatibility mode)
        print("⚠️ Using compatibility mode, manually splitting direction vectors")
        direction_df = pd.DataFrame(direction_vectors, columns=[f'dir_{i}' for i in range(6)])
        train_indices = X_train_df.index.tolist()
        val_indices = X_val_df.index.tolist() if len(X_val_df) > 0 else []
        test_indices = X_test_df.index.tolist()
        
        direction_train = direction_df.iloc[train_indices].values
        direction_val = direction_df.iloc[val_indices].values if len(val_indices) > 0 else np.array([]).reshape(0, 6)
        direction_test = direction_df.iloc[test_indices].values
    
    # Handle empty validation set case
    if direction_val is None or len(direction_val) == 0:
        direction_val = np.array([]).reshape(0, 6)

    # Normalize data
    X_train, L_train = processor.fit_transform(X_train_df, L_train_df)
    X_val, L_val = processor.transform(X_val_df, L_val_df) if len(X_val_df) > 0 else (np.array([]).reshape(0, 20), np.array([]))
    X_test, L_test = processor.transform(X_test_df, L_test_df)
    
    print(f"\n✅ Decoupled prediction data preparation complete:")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Validation set: {X_val.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    print(f"   Enhanced feature dimension: {X_train.shape[1]}")
    print(f"   Direction vector dimension: {direction_train.shape[1]}")
    
    # 2. Decoupled prediction model training
    print(f"\n🎯 Starting decoupled prediction training, results saved to: {save_dir}")
    
    model, history = train_decoupled_tsai_wu_pinn(
        X_train, L_train, case_train_ids, direction_train,
        X_val, L_val, case_val_ids, direction_val,
        processor, config,
        X_test=X_test, L_test=L_test, 
        case_test_ids=case_test_ids, direction_test=direction_test,
        output_folders=save_dir
    )
    
    # 3. Save model and results
    print(f"\n💾 Saving decoupled prediction model and training results...")
    model_path, history_path, unified_dir = save_model_and_results(model, history, processor, save_dir)
    
    # 4. Plot training history
    print(f"\n📈 Plotting decoupled prediction training history...")
    history_plot_filename = get_timestamped_filename('decoupled_training_history', 'png')
    history_plot_path = os.path.join(unified_dir, 'plots', history_plot_filename)
    plot_training_history(history, save_path=history_plot_path)
    
    # Add training convergence analysis plot
    from utils import plot_training_convergence
    convergence_plot_filename = get_timestamped_filename('training_convergence', 'png')
    convergence_plot_path = os.path.join(unified_dir, 'plots', convergence_plot_filename)
    plot_training_convergence(history, save_path=convergence_plot_path)
    
    # 5. Decoupled prediction model evaluation
    print(f"\n🔍 Starting decoupled prediction comprehensive model evaluation...")
    
    # Reconstruct test set stress tensor for evaluation
    model.eval()
    device = next(model.parameters()).device
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        L_pred_norm = model(X_test_tensor).cpu().numpy().flatten()
    
    # Convert back to original scale
    L_pred_original = processor.inverse_transform_L(L_pred_norm)
    L_test_original = processor.inverse_transform_L(L_test)
    
    # Reconstruct stress tensor
    stress_reconstructed = processor.reconstruct_stress_tensor(L_pred_original, direction_test)
    
    # 6. Plot L value prediction scatter
    print(f"\n📊 Plotting L value prediction analysis...")
    from utils import plot_L_prediction_scatter, plot_L_prediction_by_case
    
    # Main L value prediction scatter plot
    scatter_plot_filename = get_timestamped_filename('L_prediction_scatter', 'png')
    scatter_plot_path = os.path.join(unified_dir, 'plots', scatter_plot_filename)
    
    L_metrics = plot_L_prediction_scatter(
        L_test_original, L_pred_original, 
        case_ids=case_test_ids.values if hasattr(case_test_ids, 'values') else case_test_ids,
        save_path=scatter_plot_path,
        title_prefix="Decoupled Prediction PINN"
    )
    
    # L value prediction analysis grouped by case
    case_plot_filename = get_timestamped_filename('L_prediction_by_case', 'png')
    case_plot_path = os.path.join(unified_dir, 'plots', case_plot_filename)
    
    case_metrics = plot_L_prediction_by_case(
        L_test_original, L_pred_original,
        case_ids=case_test_ids.values if hasattr(case_test_ids, 'values') else case_test_ids,
        save_path=case_plot_path,
        title_prefix="Decoupled Prediction PINN"
    )
    
    # New: Generate stress plane data for plotting
    print(f"\n📐 Generating stress plane data for plotting...")
    stress_plane_data_dir = os.path.join(unified_dir, 'data')  # Changed to data folder
    os.makedirs(stress_plane_data_dir, exist_ok=True)
    
    def get_stress_plane_for_case(case_id):
        """Determine stress plane to use based on case ID"""
        if case_id in [1, 9, 10]:  # These cases are mainly shear stress
            return 'σy-τxy'
        elif case_id in [2, 5, 8]:  # These cases may be sx-txy plane
            return 'σx-τxy'
        else:  # Default to sx-sy plane
            return 'σx-σy'
    
    def generate_stress_plane_data():
        """Generate stress plane data for each case, format compatible with plotting code"""
        # Load original dataset to get stress components
        try:
            original_df = pd.read_csv('datasetnew.csv')
        except FileNotFoundError:
            print("❌ Cannot load original dataset, skipping stress plane data generation")
            return
        
        # Get test set case list
        test_case_ids = case_test_ids.values if hasattr(case_test_ids, 'values') else case_test_ids
        unique_cases = np.unique(test_case_ids)
        
        print(f"   Processing {len(unique_cases)} test cases...")
        
        for case_id in unique_cases:
            print(f"   Generating stress plane data for Case {case_id}...")
            
            # Get indices of this case in test set
            case_mask = test_case_ids == case_id
            case_indices = np.where(case_mask)[0]
            
            if len(case_indices) == 0:
                continue
            
            # Get predicted and true L values for this case
            L_true_case = L_test_original[case_mask]
            L_pred_case = L_pred_original[case_mask]
            
            # Get stress information for this case from original dataset
            original_case_data = original_df[original_df['case_id'] == case_id]
            
            if len(original_case_data) == 0:
                print(f"     Warning: Case {case_id} not found in original data")
                continue
            
            # Match test set sample count
            n_test_samples = len(L_true_case)
            if len(original_case_data) >= n_test_samples:
                stress_data = original_case_data.iloc[:n_test_samples]
            else:
                # If original data is insufficient, cycle through it
                indices = np.tile(range(len(original_case_data)), 
                                (n_test_samples // len(original_case_data) + 1))[:n_test_samples]
                stress_data = original_case_data.iloc[indices]
            
            # Extract stress components
            sx_orig = stress_data['sx'].values
            sy_orig = stress_data['sy'].values
            sz_orig = stress_data['sz'].values
            txy_orig = stress_data['txy'].values
            tyz_orig = stress_data['tyz'].values
            txz_orig = stress_data['txz'].values
            
            # Calculate original stress magnitude
            L_orig = np.sqrt(sx_orig**2 + sy_orig**2 + sz_orig**2 + 
                           txy_orig**2 + tyz_orig**2 + txz_orig**2)
            L_orig_safe = np.where(L_orig > 1e-8, L_orig, 1e-8)
            
            # Calculate unit direction vectors
            dx = sx_orig / L_orig_safe
            dy = sy_orig / L_orig_safe
            dz = sz_orig / L_orig_safe
            dxy = txy_orig / L_orig_safe
            dyz = tyz_orig / L_orig_safe
            dxz = txz_orig / L_orig_safe
            
            # Calculate true stress coordinates (using true L values)
            sx_true = L_true_case * dx
            sy_true = L_true_case * dy
            sz_true = L_true_case * dz
            txy_true = L_true_case * dxy
            tyz_true = L_true_case * dyz
            txz_true = L_true_case * dxz
            
            # Calculate predicted stress coordinates (using predicted L values)
            sx_pred = L_pred_case * dx
            sy_pred = L_pred_case * dy
            sz_pred = L_pred_case * dz
            txy_pred = L_pred_case * dxy
            tyz_pred = L_pred_case * dyz
            txz_pred = L_pred_case * dxz
            
            # Determine stress plane to use for this case
            stress_plane = get_stress_plane_for_case(case_id)
            
            # Select coordinates based on stress plane
            if stress_plane == 'σy-τxy':
                x_true, y_true = sy_true, txy_true
                x_pred, y_pred = sy_pred, txy_pred
                x_label, y_label = 'σy', 'τxy'
            elif stress_plane == 'σx-τxy':
                x_true, y_true = sx_true, txy_true
                x_pred, y_pred = sx_pred, txy_pred
                x_label, y_label = 'σx', 'τxy'
            else:  # σx-σy
                x_true, y_true = sx_true, sy_true
                x_pred, y_pred = sx_pred, sy_pred
                x_label, y_label = 'σx', 'σy'
            
            # Create DataFrame, format compatible with plotting code
            case_data = pd.DataFrame({
                # Basic information
                'sample_index': range(len(L_true_case)),
                'case_id': case_id,
                
                # Failure length (format required by plotting code)
                'true_failure_length_L': L_true_case,
                'predicted_failure_length_L': L_pred_case,
                
                # Stress plane coordinates (format required by plotting code)
                'true_stress_x': x_true,
                'true_stress_y': y_true,
                'predicted_stress_x': x_pred,
                'predicted_stress_y': y_pred,
                
                # Stress plane information
                'stress_plane': stress_plane,
                'x_label': x_label,
                'y_label': y_label,
                
                # All stress components (complete data)
                'sx_true': sx_true,
                'sy_true': sy_true,
                'sz_true': sz_true,
                'txy_true': txy_true,
                'tyz_true': tyz_true,
                'txz_true': txz_true,
                
                'sx_pred': sx_pred,
                'sy_pred': sy_pred,
                'sz_pred': sz_pred,
                'txy_pred': txy_pred,
                'tyz_pred': tyz_pred,
                'txz_pred': txz_pred,
                
                # Original stress components (reference)
                'sx_original': sx_orig,
                'sy_original': sy_orig,
                'sz_original': sz_orig,
                'txy_original': txy_orig,
                'tyz_original': tyz_orig,
                'txz_original': txz_orig,
                
                # Calculated relative error
                'L_relative_error': np.abs(L_pred_case - L_true_case) / (L_true_case + 1e-8) * 100
            })
            
            # Save to CSV file, using filename format expected by plotting code
            case_filename = f'case_{case_id}_detailed_results.csv'
            case_filepath = os.path.join(stress_plane_data_dir, case_filename)
            case_data.to_csv(case_filepath, index=False)
            
            print(f"     ✅ Case {case_id}: {len(case_data)} samples, stress plane {stress_plane}")
            print(f"        Saved to: {case_filename}")
            print(f"        Stress coordinate range: X[{x_true.min():.1f}, {x_true.max():.1f}], Y[{y_true.min():.1f}, {y_true.max():.1f}]")
    
    # Execute stress plane data generation
    generate_stress_plane_data()
    
    # Generate stress plane data summary information
    summary_info = {
        'generation_time': get_timestamped_filename('', '')[1:-1],
        'total_test_samples': len(L_test_original),
        'unique_cases': len(np.unique(case_test_ids.values if hasattr(case_test_ids, 'values') else case_test_ids)),
        'data_format': 'case_{case_id}_detailed_results.csv',
        'data_location': stress_plane_data_dir,
        'stress_plane_mapping': {
            'σx-σy_cases': [int(c) for c in np.unique(case_test_ids.values if hasattr(case_test_ids, 'values') else case_test_ids) 
                           if get_stress_plane_for_case(c) == 'σx-σy'],
            'σy-τxy_cases': [int(c) for c in np.unique(case_test_ids.values if hasattr(case_test_ids, 'values') else case_test_ids) 
                            if get_stress_plane_for_case(c) == 'σy-τxy'],
            'σx-τxy_cases': [int(c) for c in np.unique(case_test_ids.values if hasattr(case_test_ids, 'values') else case_test_ids) 
                            if get_stress_plane_for_case(c) == 'σx-τxy']
        },
        'usage_note': 'Data format fully compatible with plotting scripts, ready for stress plane visualization',
        'coordinate_info': {
            'σx-σy': 'Normal stress plane for biaxial loading cases',
            'σy-τxy': 'Normal-shear stress plane for shear-dominated cases (1,9,10)',
            'σx-τxy': 'Normal-shear stress plane for specific cases (2,5,8)'
        }
    }
    
    summary_filepath = os.path.join(stress_plane_data_dir, 'stress_plane_data_summary.json')
    import json
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        json.dump(summary_info, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✅ Stress plane data generation complete:")
    print(f"   Data directory: {stress_plane_data_dir}")
    print(f"   File format: case_{{case_id}}_detailed_results.csv")
    print(f"   Summary info: {summary_filepath}")
    print(f"   Stress plane distribution:")
    for plane, cases in summary_info['stress_plane_mapping'].items():
        if cases:
            print(f"     {plane}: Cases {cases}")
    print(f"   🎨 Data ready for plotting scripts!")
    
    # 7. Comprehensive evaluation report
    print(f"🔍 Decoupled prediction evaluation results:")
    
    from sklearn.metrics import mean_squared_error, r2_score
    
    # L value prediction performance
    L_mse = mean_squared_error(L_test_original, L_pred_original)
    L_rmse = np.sqrt(L_mse)
    L_r2 = r2_score(L_test_original, L_pred_original)
    relative_error = np.abs(L_pred_original - L_test_original) / (L_test_original + 1e-8) * 100
    
    print(f"📊 Failure length L prediction performance:")
    print(f"   R² = {L_r2:.4f} (target: ≥0.91)")
    print(f"   RMSE = {L_rmse:.4f} MPa (target: ≤135)")
    print(f"   Average relative error = {relative_error.mean():.2f}% ± {relative_error.std():.2f}%")
    print(f"   Median relative error = {np.median(relative_error):.2f}%")
    print(f"   L value prediction range: [{L_pred_original.min():.2f}, {L_pred_original.max():.2f}] MPa")
    print(f"   L value true range: [{L_test_original.min():.2f}, {L_test_original.max():.2f}] MPa")
    
    # Performance assessment
    if L_r2 >= 0.91:
        print(f"🎉 Performance meets target: R² = {L_r2:.4f} ≥ 0.91")
    else:
        print(f"⚠️  Performance below expectation: R² = {L_r2:.4f} < 0.91")
        
    if L_rmse <= 135:
        print(f"🎉 Error meets target: RMSE = {L_rmse:.1f} ≤ 135 MPa")
    else:
        print(f"⚠️  Error too high: RMSE = {L_rmse:.1f} > 135 MPa")
    
    print(f"\n🔄 Stress reconstruction performance:")
    print(f"   Reconstructed stress range: [{stress_reconstructed.min():.2f}, {stress_reconstructed.max():.2f}] MPa")
    
    # Calculate reconstructed stress L value for comparison
    L_reconstructed = np.sqrt(np.sum(stress_reconstructed**2, axis=1))
    L_r2_reconstructed = r2_score(L_test_original, L_reconstructed)
    reconstruction_error = np.abs(L_reconstructed - L_test_original) / (L_test_original + 1e-8) * 100
    print(f"   Reconstructed stress L value R² = {L_r2_reconstructed:.4f}")
    print(f"   Reconstruction consistency error = {reconstruction_error.mean():.4f}% ± {reconstruction_error.std():.4f}%")
    
    # Analysis results by case
    print(f"\n📋 Analysis by case:")
    for case_id, metrics in case_metrics.items():
        print(f"   Case {case_id}: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.1f} MPa, "
              f"Rel. error={metrics['mean_relative_error']:.1f}%, Samples={metrics['samples']}")
    
    # 8. Save detailed evaluation report
    report_filename = get_timestamped_filename('evaluation_report', 'txt')
    report_path = os.path.join(unified_dir, 'reports', report_filename)
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Decoupled Prediction Tsai-Wu PINN Model Evaluation Report\n")
        f.write("="*50 + "\n")
        f.write(f"Report generation time: {get_timestamped_filename('', '')[1:-1]}\n")
        f.write(f"Configuration type: {'Optimal' if config == OPTIMAL_CONFIG else 'Standard'}\n\n")
        
        f.write("Model configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("Performance targets (based on hyperparameter search):\n")
        f.write("  Target R² ≥ 0.91\n")
        f.write("  Target RMSE ≤ 135 MPa\n")
        f.write("  Target relative error ≤ 35%\n\n")
        
        f.write("L value prediction performance:\n")
        f.write(f"  R² = {L_r2:.6f} {'✓' if L_r2 >= 0.91 else '✗'}\n")
        f.write(f"  RMSE = {L_rmse:.4f} MPa {'✓' if L_rmse <= 135 else '✗'}\n")
        f.write(f"  Average relative error = {relative_error.mean():.2f}% {'✓' if relative_error.mean() <= 35 else '✗'}\n")
        f.write(f"  Relative error std = {relative_error.std():.2f}%\n")
        f.write(f"  Median relative error = {np.median(relative_error):.2f}%\n")
        f.write(f"  Min relative error = {relative_error.min():.2f}%\n")
        f.write(f"  Max relative error = {relative_error.max():.2f}%\n\n")
        
        f.write("Stress reconstruction performance:\n")
        f.write(f"  Reconstructed L value R² = {L_r2_reconstructed:.6f}\n")
        f.write(f"  Reconstruction consistency error = {reconstruction_error.mean():.4f}%\n\n")
        
        f.write("Analysis by case:\n")
        for case_id, metrics in case_metrics.items():
            f.write(f"  Case {case_id}:\n")
            f.write(f"    R² = {metrics['r2']:.6f}\n")
            f.write(f"    RMSE = {metrics['rmse']:.4f} MPa\n")
            f.write(f"    Average relative error = {metrics['mean_relative_error']:.2f}%\n")
            f.write(f"    Sample count = {metrics['samples']}\n\n")
    
    print(f"Detailed evaluation report saved: {report_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("Usage:")
        print("  python main.py              # Standard training mode")
        print("  python main.py --test       # Test mode")
        print("  python main.py --optimal    # Optimal configuration mode (recommended)")
        print("  python main.py --search     # Hyperparameter search mode")
    else:
        main()
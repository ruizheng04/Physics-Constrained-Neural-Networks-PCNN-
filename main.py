# main.py
# Responsibility: Program entry point, connects all modules, executes complete workflow.

import sys
import warnings
import pandas as pd
import os
import torch
import numpy as np

# Add global warning filters to completely suppress sklearn feature_names warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', module='sklearn')

# Import from various modules
from config import LARC_MATERIAL_PROPERTIES
from data_processor import TsaiWuPINNDataProcessor
from trainer import train_decoupled_tsai_wu_pinn
from utils import setup_plotting, plot_training_history, save_model_and_results, get_timestamped_filename

# Optimal configuration - based on hyperparameter search results
CONFIG = {
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
    
    # Physics constraint weights - fully adaptive strategy
    'lambda_final': 5,
    'W_physics': 2.0,
    'adaptive_balance': True,
    'gradient_balance_alpha': 0.1,
    'physics_warmup_rate': 0.05,
    
    # Data processing
    'normalization_method': 'minmax_columnwise',
    'batch_size': 64,
    
    # Regularization
    'l1_lambda': 0.0,
    'l2_lambda': 0.0
}

def main():
    """Main function - Decoupled prediction version of LaRC criterion constrained PINN"""
    print("\n" + "="*90)
    print("🚀 Decoupled Prediction LaRC-PINN Model (Adaptive Weight)")
    print("📋 Input: 20D Enhanced Features (14 Material + 6 Direction)")
    print("📊 Output: 1D Failure Length L")
    print("🔧 Core Strategy: Predict L → Reconstruct Stress → Validate Physics")
    print("⚖️  Physics Constraints: LaRC Failure Criterion on Reconstructed Stress")
    print("🎯 Weight Strategy: Fully Adaptive Gradient-Based Balancing")
    print("📈 Adaptive Parameters: α={}, warmup_rate={}".format(
        CONFIG['gradient_balance_alpha'], CONFIG['physics_warmup_rate']))
    print("="*90)
    
    setup_plotting()
    # Set stricter warning filters
    warnings.filterwarnings('ignore')
    
    # Use optimal configuration
    config = CONFIG
    save_dir = 'results_optimal_larc'
    
    print(f"🔧 Using adaptive weight mechanism:")
    print(f"   Architecture: {config['layer_sizes']}")
    print(f"   Activation: {config['activation']}")
    print(f"   Learning rate: {config['lr']}")
    print(f"   Dropout rate: {config['dropout_rate']}")
    print(f"   ⚖️  Adaptive balance: ENABLED")
    print(f"   📊 Gradient balance coefficient α: {config['gradient_balance_alpha']}")
    print(f"   🔄 Physics warmup rate: {config['physics_warmup_rate']}")
    print(f"   ❌ Fixed physics weight: DISABLED (W_physics={config.get('W_physics', 0.0)})")

    print(f"⚖️  Weight strategy: Fully adaptive (dynamic balancing based on gradients)")
    print(f"🧠 Activation function: {config.get('activation', 'swiglu')}")
    print("✅ Decoupled prediction: 20D features → 1D L value → 6D stress reconstruction")
    print("🔕 Silent mode enabled, reduced sklearn warning messages")

    # 1. Decoupled prediction data processing
    processor = TsaiWuPINNDataProcessor()
    # Enable silent mode to reduce output
    processor.verbose = False
    
    try:
        df = processor.load_data('datasetnew.csv')
    except FileNotFoundError:
        print("❌ Error: 'datasetnew.csv' not found. Please ensure data file is in current directory.")
        return
    
    # Run LaRC parameter validation (only on first run)
    print("\n🔧 Running LaRC parameter and data validation...")
    try:
        from debug_larc import analyze_dataset_stress_distribution, verify_larc_parameters
        analyze_dataset_stress_distribution()
        verify_larc_parameters()
    except Exception as e:
        print(f"⚠️ LaRC validation tool error: {e}")
        
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
    
    # Handle empty validation set
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
    
    # 5. Decoupled prediction model evaluation (moved forward)
    print(f"\n🔍 Starting comprehensive decoupled prediction model evaluation...")
    
    # Predict on all datasets
    model.eval()
    device = next(model.parameters()).device
    
    # Training set predictions
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    with torch.no_grad():
        L_pred_train_norm = model(X_train_tensor).cpu().numpy().flatten()
    L_pred_train_original = processor.inverse_transform_L(L_pred_train_norm)
    L_train_original = processor.inverse_transform_L(L_train)
    stress_train_reconstructed = processor.reconstruct_stress_tensor(L_pred_train_original, direction_train)
    
    # Validation set predictions (if exists)
    if len(X_val) > 0:
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        with torch.no_grad():
            L_pred_val_norm = model(X_val_tensor).cpu().numpy().flatten()
        L_pred_val_original = processor.inverse_transform_L(L_pred_val_norm)
        L_val_original = processor.inverse_transform_L(L_val)
        stress_val_reconstructed = processor.reconstruct_stress_tensor(L_pred_val_original, direction_val)
    else:
        L_pred_val_original = np.array([])
        L_val_original = np.array([])
        stress_val_reconstructed = np.array([]).reshape(0, 6)
    
    # Test set predictions
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        L_pred_test_norm = model(X_test_tensor).cpu().numpy().flatten()
    L_pred_test_original = processor.inverse_transform_L(L_pred_test_norm)
    L_test_original = processor.inverse_transform_L(L_test)
    stress_test_reconstructed = processor.reconstruct_stress_tensor(L_pred_test_original, direction_test)
    
    # 6. Generate detailed CSV result files (including all datasets)
    print(f"\n📊 Generating complete dataset prediction results CSV files...")
    from utils import (generate_detailed_case_results_csv, 
                       generate_unified_test_results_csv,
                       generate_complete_dataset_results_csv)
    
    # Generate unified results CSV for complete dataset (new)
    complete_csv_path = generate_complete_dataset_results_csv(
        L_train_original, L_pred_train_original,
        case_train_ids.values if hasattr(case_train_ids, 'values') else case_train_ids,
        direction_train, processor,
        L_val_original, L_pred_val_original,
        case_val_ids.values if hasattr(case_val_ids, 'values') else case_val_ids if len(X_val) > 0 else np.array([]),
        direction_val,
        L_test_original, L_pred_test_original,
        case_test_ids.values if hasattr(case_test_ids, 'values') else case_test_ids,
        direction_test,
        unified_dir
    )
    
    # Generate detailed results CSV grouped by case (test set only)
    case_csv_dir = generate_detailed_case_results_csv(
        L_test_original, L_pred_test_original, 
        case_test_ids.values if hasattr(case_test_ids, 'values') else case_test_ids,
        direction_test, processor, unified_dir
    )
    
    # Generate unified test results CSV
    unified_csv_path = generate_unified_test_results_csv(
        L_test_original, L_pred_test_original,
        case_test_ids.values if hasattr(case_test_ids, 'values') else case_test_ids,
        direction_test, processor, unified_dir
    )
    
    print(f"📁 Complete dataset results: {complete_csv_path}")
    print(f"📁 Case-grouped results: {case_csv_dir}")
    print(f"📁 Test set results: {unified_csv_path}")
    
    # 7. Plot L value prediction scatter plots (including train/test comparison)
    print(f"\n📊 Plotting L value prediction analysis...")
    from utils import (plot_L_prediction_scatter, 
                       plot_L_prediction_by_case,
                       plot_train_test_comparison)
    
    # Train/test comparison plot (new)
    comparison_plot_filename = get_timestamped_filename('train_test_comparison', 'png')
    comparison_plot_path = os.path.join(unified_dir, 'plots', comparison_plot_filename)
    
    plot_train_test_comparison(
        L_train_original, L_pred_train_original,
        L_test_original, L_pred_test_original,
        save_path=comparison_plot_path,
        title_prefix="Decoupled Prediction PINN"
    )
    
    # Main L value prediction scatter plot (test set)
    scatter_plot_filename = get_timestamped_filename('L_prediction_scatter', 'png')
    scatter_plot_path = os.path.join(unified_dir, 'plots', scatter_plot_filename)
    
    L_metrics = plot_L_prediction_scatter(
        L_test_original, L_pred_test_original, 
        case_ids=case_test_ids.values if hasattr(case_test_ids, 'values') else case_test_ids,
        save_path=scatter_plot_path,
        title_prefix="Decoupled Prediction PINN"
    )
    
    # L value prediction analysis grouped by case
    case_plot_filename = get_timestamped_filename('L_prediction_by_case', 'png')
    case_plot_path = os.path.join(unified_dir, 'plots', case_plot_filename)
    
    case_metrics = plot_L_prediction_by_case(
        L_test_original, L_pred_test_original,
        case_ids=case_test_ids.values if hasattr(case_test_ids, 'values') else case_test_ids,
        save_path=case_plot_path,
        title_prefix="Decoupled Prediction PINN"
    )
    
    # 8. Comprehensive evaluation report (added training set evaluation)
    print(f"🔍 Decoupled prediction evaluation results:")
    
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Training set performance (new)
    L_train_mse = mean_squared_error(L_train_original, L_pred_train_original)
    L_train_rmse = np.sqrt(L_train_mse)
    L_train_r2 = r2_score(L_train_original, L_pred_train_original)
    train_relative_error = np.abs(L_pred_train_original - L_train_original) / (L_train_original + 1e-8) * 100
    
    print(f"\n📊 Training set performance:")
    print(f"   R² = {L_train_r2:.4f}")
    print(f"   RMSE = {L_train_rmse:.4f} MPa")
    print(f"   Mean relative error = {train_relative_error.mean():.2f}% ± {train_relative_error.std():.2f}%")
    
    # Test set performance
    L_mse = mean_squared_error(L_test_original, L_pred_test_original)
    L_rmse = np.sqrt(L_mse)
    L_r2 = r2_score(L_test_original, L_pred_test_original)
    relative_error = np.abs(L_pred_test_original - L_test_original) / (L_test_original + 1e-8) * 100
    
    print(f"\n📊 Test set failure length L prediction performance:")
    print(f"   R² = {L_r2:.4f} (target: ≥0.91)")
    print(f"   RMSE = {L_rmse:.4f} MPa (target: ≤135)")
    print(f"   Mean relative error = {relative_error.mean():.2f}% {'✓' if relative_error.mean() <= 35 else '✗'}")
    print(f"   Median relative error = {np.median(relative_error):.2f}%")
    print(f"   L value prediction range: [{L_pred_test_original.min():.2f}, {L_pred_test_original.max():.2f}] MPa")
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
    print(f"   Reconstructed stress range: [{stress_test_reconstructed.min():.2f}, {stress_test_reconstructed.max():.2f}] MPa")
    
    # Calculate L value from reconstructed stress for comparison
    L_reconstructed = np.sqrt(np.sum(stress_test_reconstructed**2, axis=1))
    L_r2_reconstructed = r2_score(L_test_original, L_reconstructed)
    reconstruction_error = np.abs(L_reconstructed - L_test_original) / (L_test_original + 1e-8) * 100
    print(f"   Reconstructed stress L value R² = {L_r2_reconstructed:.4f}")
    print(f"   Reconstruction consistency error = {reconstruction_error.mean():.4f}% ± {reconstruction_error.std():.4f}%")
    
    # Case-by-case analysis results
    print(f"\n📋 Case-by-case analysis results:")
    for case_id, metrics in case_metrics.items():
        print(f"   Case {case_id}: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.1f} MPa, "
              f"Relative error={metrics['mean_relative_error']:.1f}%, Samples={metrics['samples']}")
    
    # 9. Save detailed evaluation report (updated content)
    report_filename = get_timestamped_filename('evaluation_report', 'txt')
    report_path = os.path.join(unified_dir, 'reports', report_filename)
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Decoupled Prediction LaRC-PINN Model Evaluation Report\n")
        f.write("="*50 + "\n")
        f.write(f"Report generation time: {get_timestamped_filename('', '')[1:-1]}\n")
        f.write(f"Configuration type: Optimal configuration (LaRC criterion)\n")
        f.write(f"Complete dataset results: {complete_csv_path}\n")
        f.write(f"CSV data file location: {case_csv_dir}\n")
        f.write(f"Unified results file: {unified_csv_path}\n\n")
        
        f.write("Model configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("Dataset split:\n")
        f.write(f"  Training set samples: {len(L_train_original)}\n")
        f.write(f"  Validation set samples: {len(L_val_original)}\n")
        f.write(f"  Test set samples: {len(L_test_original)}\n")
        f.write(f"  Total samples: {len(L_train_original) + len(L_val_original) + len(L_test_original)}\n\n")
        
        f.write("Performance targets (based on hyperparameter search):\n")
        f.write("  Target R² ≥ 0.91\n")
        f.write("  Target RMSE ≤ 135 MPa\n")
        f.write("  Target relative error ≤ 35%\n\n")
        
        f.write("Training set performance:\n")
        f.write(f"  R² = {L_train_r2:.6f}\n")
        f.write(f"  RMSE = {L_train_rmse:.4f} MPa\n")
        f.write(f"  Mean relative error = {train_relative_error.mean():.2f}%\n")
        f.write(f"  Relative error std = {train_relative_error.std():.2f}%\n")
        f.write(f"  Median relative error = {np.median(train_relative_error):.2f}%\n\n")
        
        f.write("Test set L value prediction performance:\n")
        f.write(f"  R² = {L_r2:.6f} {'✓' if L_r2 >= 0.91 else '✗'}\n")
        f.write(f"  RMSE = {L_rmse:.4f} MPa {'✓' if L_rmse <= 135 else '✗'}\n")
        f.write(f"  Mean relative error = {relative_error.mean():.2f}% {'✓' if relative_error.mean() <= 35 else '✗'}\n")
        f.write(f"  Relative error std = {relative_error.std():.2f}%\n")
        f.write(f"  Median relative error = {np.median(relative_error):.2f}%\n")
        f.write(f"  Min relative error = {relative_error.min():.2f}%\n")
        f.write(f"  Max relative error = {relative_error.max():.2f}%\n\n")
        
        f.write("Stress reconstruction performance:\n")
        f.write(f"  Reconstructed L value R² = {L_r2_reconstructed:.6f}\n")
        f.write(f"  Reconstruction consistency error = {reconstruction_error.mean():.4f}%\n\n")
        
        f.write("Case-by-case analysis:\n")
        for case_id, metrics in case_metrics.items():
            f.write(f"  Case {case_id}:\n")
            f.write(f"    R² = {metrics['r2']:.6f}\n")
            f.write(f"    RMSE = {metrics['rmse']:.4f} MPa\n")
            f.write(f"    Mean relative error = {metrics['mean_relative_error']:.2f}%\n")
            f.write(f"    Samples = {metrics['samples']}\n\n")
        
        f.write("Data file description:\n")
        f.write(f"  Complete dataset results: {complete_csv_path}\n")
        f.write(f"    - Contains all prediction results for train/val/test sets\n")
        f.write(f"    - 'dataset_type' column identifies data source (train/val/test)\n")
        f.write(f"  Detailed case results: {case_csv_dir}/case_*_detailed_results_corrected.csv\n")
        f.write(f"  Unified test results: {unified_csv_path}\n")
        f.write(f"  Data format ready for plotting scripts\n\n")
    
    print(f"Detailed evaluation report saved: {report_path}")
    
    # 10. Generate plotting instructions (updated description)
    print(f"\n🎨 Plotting data preparation complete!")
    print(f"📊 Complete dataset results: {complete_csv_path}")
    print(f"   - Contains all predictions for train/val/test sets")
    print(f"   - 'dataset_type' column distinguishes data source")
    print(f"📊 Case-grouped data: {case_csv_dir}")
    print(f"📊 Original dataset: datasetnew.csv")
    print(f"📊 Use the following command to run plotting script:")
    print(f"   python 绘图.py")
    print(f"   (Please ensure paths in 绘图.py point to: {case_csv_dir})")

if __name__ == "__main__":
    main()
# trainer.py
# Responsibility: Contains core logic for model training and validation.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd

# Import necessary classes and functions from other modules
from model import TsaiWuPINN
from physics_loss import LaRCPhysicsLoss
from config import LARC_MATERIAL_PROPERTIES

def train_decoupled_tsai_wu_pinn(X_train, L_train, case_train_ids, direction_train,
                                X_val, L_val, case_val_ids, direction_val,
                                processor, config,
                                X_test=None, L_test=None, case_test_ids=None, direction_test=None,
                                output_folders='results_optimal_larc'
):
    """Decoupled prediction version LaRC-PINN training function - fully adaptive weight mechanism"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Decoupled prediction training mode started (adaptive weights)")
    print(f"Training device: {device}")
    
    # Extract configuration parameters
    layer_sizes = config['layer_sizes']
    activation = config['activation']
    lr = config['lr']
    dropout_rate = config['dropout_rate']
    weight_decay = config['weight_decay']
    lambda_final = config['lambda_final']
    physics_delay_epochs = config['physics_delay_epochs']
    warm_start_epochs = config['warm_start_epochs']
    W_physics = config.get('W_physics', 0.0)
    
    # Extract physics constraint related parameters
    lambda_final = config.get('lambda_final', 1.0)
    W_physics = config.get('W_physics', 0.0)
    adaptive_balance = config.get('adaptive_balance', True)
    gradient_balance_alpha = config.get('gradient_balance_alpha', 0.1)
    physics_warmup_rate = config.get('physics_warmup_rate', 0.05)
    
    # Check if adaptive mechanism is enabled
    use_adaptive = adaptive_balance and W_physics == 0.0
    
    if use_adaptive:
        print(f"⚖️  Fully adaptive weight mechanism enabled:")
        print(f"   Gradient balance coefficient α = {gradient_balance_alpha}")
        print(f"   Physics warmup rate = {physics_warmup_rate}")
        print(f"   Fixed weight W_physics = {W_physics} (disabled)")
    else:
        print(f"⚠️  Using fixed weight mode: W_physics = {W_physics}")
    
    # Get verbose parameter, default to True
    verbose = config.get('verbose', True)
    
    # Initialize physics weight (important: must be initialized before training loop)
    current_lambda = 0.0  # Start from 0
    
    # Use passed parameters or default values
    max_epochs = 3000
    early_stop_patience = 500
    checkpoint_epoch = max(500, max_epochs // 6) if output_folders else max_epochs + 1  # Disable checkpoint output
    
    print(f"\n📊 Decoupled prediction training data validation:")
    print(f"  Enhanced features X: {X_train.shape} (should be N×20)")
    print(f"  Failure length L: {L_train.shape if hasattr(L_train, 'shape') else len(L_train)} (should be N×1 or N)")
    print(f"  Stress direction: {direction_train.shape} (should be N×6)")
    print(f"  Case IDs: {len(case_train_ids)}")
    
    if X_train.shape[1] != 20:
        raise ValueError(f"Enhanced feature dimension should be 20, actual is {X_train.shape[1]}")
    if direction_train.shape[1] != 6:
        raise ValueError(f"Stress direction dimension should be 6, actual is {direction_train.shape[1]}")
    
    # Create decoupled prediction model
    print(f"\n🏗️ Building decoupled prediction model:")
    model = TsaiWuPINN(
        layer_sizes=layer_sizes,
        dropout_rate=dropout_rate,
        activation=activation
    ).to(device)
    
    # Optimizer and scheduler - use more conservative learning rate
    initial_lr = lr * 0.5  # Further reduce initial learning rate
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=max(50, early_stop_patience//3), min_lr=1e-7)
    
    # Physics loss calculator
    physics_calculator = LaRCPhysicsLoss(LARC_MATERIAL_PROPERTIES, device, processor)
    
    # Convert data to tensors - enhanced data checking
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    
    # Handle different formats of L_train - enhanced compatibility
    if isinstance(L_train, pd.Series):
        L_train_array = L_train.values
    elif isinstance(L_train, np.ndarray):
        L_train_array = L_train
    else:
        L_train_array = np.array(L_train)
    
    L_train_tensor = torch.FloatTensor(L_train_array).to(device)
    if L_train_tensor.ndim == 1:
        L_train_tensor = L_train_tensor.unsqueeze(1)  # Ensure column vector
    
    # Validate direction vector validity
    if direction_train is None or len(direction_train) == 0:
        raise ValueError("Training direction vectors cannot be empty")
    
    direction_train_tensor = torch.FloatTensor(direction_train).to(device)
    
    # Validate data dimension consistency
    if X_train_tensor.shape[0] != L_train_tensor.shape[0] or X_train_tensor.shape[0] != direction_train_tensor.shape[0]:
        raise ValueError(f"Data dimensions inconsistent: X({X_train_tensor.shape[0]}) vs L({L_train_tensor.shape[0]}) vs Direction({direction_train_tensor.shape[0]})")
    
    print(f"\n🔍 Training data numerical check:")
    print(f"  X_train range: [{X_train_tensor.min().item():.4f}, {X_train_tensor.max().item():.4f}]")
    print(f"  L_train range: [{L_train_tensor.min().item():.4f}, {L_train_tensor.max().item():.4f}]")
    print(f"  direction_train range: [{direction_train_tensor.min().item():.4f}, {direction_train_tensor.max().item():.4f}]")
    print(f"  Data batch size: {X_train_tensor.shape[0]}")

    # Handle case_train_ids
    if isinstance(case_train_ids, pd.Series):
        case_train_array = case_train_ids.values
    else:
        case_train_array = np.array(case_train_ids)
    case_train_tensor = torch.FloatTensor(case_train_array).to(device)
    
    # Handle validation set data
    if len(X_val) > 0:
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        
        if isinstance(L_val, pd.Series):
            L_val_array = L_val.values
        elif isinstance(L_val, np.ndarray):
            L_val_array = L_val
        else:
            L_val_array = np.array(L_val)
        
        L_val_tensor = torch.FloatTensor(L_val_array).to(device)
        if L_val_tensor.ndim == 1:
            L_val_tensor = L_val_tensor.unsqueeze(1)
        
        direction_val_tensor = torch.FloatTensor(direction_val).to(device)
        
        if isinstance(case_val_ids, pd.Series):
            case_val_array = case_val_ids.values
        else:
            case_val_array = np.array(case_val_ids) if len(case_val_ids) > 0 else np.array([])
        case_val_tensor = torch.FloatTensor(case_val_array).to(device) if len(case_val_array) > 0 else torch.empty(0).to(device)
    else:
        # Create empty validation tensors
        X_val_tensor = torch.empty(0, X_train.shape[1]).to(device)
        L_val_tensor = torch.empty(0, 1).to(device)
        direction_val_tensor = torch.empty(0, 6).to(device)
        case_val_tensor = torch.empty(0).to(device)
    
    # Training history (add adaptive weight recording)
    history = {
        'train_loss': [], 'val_loss': [], 'test_loss': [],
        'train_data_loss': [], 'train_physics_loss': [],
        'val_data_loss': [], 'val_physics_loss': [],
        'test_data_loss': [], 'test_physics_loss': [],
        'train_r2': [], 'val_r2': [], 'test_r2': [],
        'learning_rate': [],
        'physics_weight': [],
        'adaptive_weight': [],  # New: record adaptive weight
        'epoch': []
    }
    
    print(f"\n🎯 Decoupled prediction training strategy:")
    print(f"📊 Phase 1 (Pure L value fitting): Epochs 1-{warm_start_epochs}")
    print(f"🔧 Phase 2 (Progressive physics constraint): Epochs {warm_start_epochs+1}-{physics_delay_epochs}")
    print(f"⚖️  Phase 3 (L value + physics joint optimization): Epochs {physics_delay_epochs+1}-{max_epochs}")
    print(f"🎚️  Physics weight: λ dynamically adjusted (0→{lambda_final}), W_physics={W_physics}, Adaptive balance={'Enabled' if adaptive_balance else 'Disabled'}")
    print("="*70)

    # Early stopping mechanism
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = early_stop_patience

    for epoch in range(1, max_epochs + 1):
        model.train()
        
        # Dynamically adjust base lambda for physics loss
        if epoch <= warm_start_epochs:
            current_lambda = 0.0
        elif epoch <= physics_delay_epochs:
            progress = (epoch - warm_start_epochs) / (physics_delay_epochs - warm_start_epochs)
            current_lambda = lambda_final * progress * physics_warmup_rate
        else:
            current_lambda = lambda_final

        # === Core training step ===
        optimizer.zero_grad()
        
        L_pred = model(X_train_tensor)
        stress_reconstructed = L_pred * direction_train_tensor
        
        # Data loss
        loss_L = nn.MSELoss()(L_pred, L_train_tensor)
        
        # Physics loss - use adaptive or fixed weight
        if current_lambda > 0:
            dummy_true_stress = torch.zeros_like(stress_reconstructed)
            
            if use_adaptive:
                # Use adaptive weight mechanism
                try:
                    total_loss, loss_data_computed, loss_physics, adaptive_weight, loss_L_monitor = \
                        physics_calculator.compute_adaptive_normalized_loss(
                            stress_reconstructed, 
                            dummy_true_stress,
                            case_train_tensor, 
                            current_lambda, 
                            alpha=gradient_balance_alpha
                        )
                    
                    # Numerical health check
                    if torch.isnan(loss_physics) or torch.isinf(loss_physics):
                        loss_physics = torch.tensor(0.0, device=device)
                        adaptive_weight = torch.tensor(1.0, device=device)
                        total_loss = loss_L
                    
                except Exception as e:
                    print(f"⚠️ Adaptive weight calculation failed (epoch {epoch}): {e}")
                    loss_physics = torch.tensor(0.0, device=device)
                    adaptive_weight = torch.tensor(1.0, device=device)
                    total_loss = loss_L
                    loss_L_monitor = torch.tensor(0.0, device=device)
                
                effective_physics_weight = current_lambda * adaptive_weight.item()
                
            else:
                # Use fixed weight mode
                try:
                    total_loss, loss_data_computed, loss_physics, loss_L_monitor = \
                        physics_calculator.compute_all_normalized_loss(
                            stress_reconstructed, 
                            dummy_true_stress,
                            case_train_tensor, 
                            current_lambda, 
                            W_physics
                        )
                    adaptive_weight = torch.tensor(W_physics, device=device)
                    
                except Exception as e:
                    print(f"⚠️ Fixed weight calculation failed (epoch {epoch}): {e}")
                    loss_physics = torch.tensor(0.0, device=device)
                    adaptive_weight = torch.tensor(W_physics, device=device)
                    total_loss = loss_L
                    loss_L_monitor = torch.tensor(0.0, device=device)
                
                effective_physics_weight = current_lambda * W_physics
        else:
            loss_physics = torch.tensor(0.0, device=device)
            adaptive_weight = torch.tensor(0.0, device=device)
            total_loss = loss_L
            loss_L_monitor = torch.tensor(0.0, device=device)
            effective_physics_weight = 0.0
        
        # Backpropagation
        if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() > 1e6:
            print(f"⚠️ Epoch {epoch}: Loss abnormal ({total_loss.item():.2e}), skipping")
            continue
        
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # === Validation step ===
        model.eval()
        with torch.no_grad():
            if len(X_val) > 0:
                L_val_pred = model(X_val_tensor)
                stress_val_reconstructed = L_val_pred * direction_val_tensor
                val_loss_L = nn.MSELoss()(L_val_pred, L_val_tensor)
                
                if current_lambda > 0:
                    dummy_val_true_stress = torch.zeros_like(stress_val_reconstructed)
                    try:
                        if use_adaptive:
                            val_total_loss, _, val_loss_physics, val_adaptive_weight, _ = \
                                physics_calculator.compute_adaptive_normalized_loss(
                                    stress_val_reconstructed,
                                    dummy_val_true_stress,
                                    case_val_tensor,
                                    current_lambda,
                                    alpha=gradient_balance_alpha
                                )
                        else:
                            val_total_loss, _, val_loss_physics, _ = \
                                physics_calculator.compute_all_normalized_loss(
                                    stress_val_reconstructed,
                                    dummy_val_true_stress,
                                    case_val_tensor,
                                    current_lambda,
                                    W_physics
                                )
                    except:
                        val_loss_physics = torch.tensor(0.0, device=device)
                        val_total_loss = val_loss_L
                else:
                    val_loss_physics = torch.tensor(0.0, device=device)
                    val_total_loss = val_loss_L
            else:
                val_loss_L = loss_L
                val_loss_physics = loss_physics
                val_total_loss = total_loss
        
        # Learning rate scheduling and early stopping
        scheduler.step(val_total_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if val_total_loss < best_val_loss * 0.995:
            best_val_loss = val_total_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        current_patience_limit = max_patience * 2 if epoch <= warm_start_epochs else max_patience
        
        if patience_counter > current_patience_limit and epoch > warm_start_epochs:
            print(f"🛑 Early stopping: Validation loss not improved for {patience_counter} epochs")
            break
        
        # Record history
        history['train_loss'].append(total_loss.item())
        history['val_loss'].append(val_total_loss.item())
        history['train_data_loss'].append(loss_L.item())
        history['train_physics_loss'].append(loss_physics.item())
        history['val_data_loss'].append(val_loss_L.item())
        history['val_physics_loss'].append(val_loss_physics.item())
        history['learning_rate'].append(current_lr)
        history['physics_weight'].append(effective_physics_weight)
        history['adaptive_weight'].append(adaptive_weight.item() if isinstance(adaptive_weight, torch.Tensor) else adaptive_weight)
        history['epoch'].append(epoch)
        
        # Print progress
        print_interval = 100 if config.get('verbose', True) else 1000
        if epoch % print_interval == 0 or epoch == 1:
            with torch.no_grad():
                L_pred_train_np = L_pred.cpu().numpy()
                L_train_np = L_train_tensor.cpu().numpy()
                from sklearn.metrics import r2_score
                train_r2 = r2_score(L_train_np, L_pred_train_np)

            weight_info = f"Adaptive W={adaptive_weight.item():.4f}" if use_adaptive else f"Fixed W={W_physics:.2f}"
            print(f"Epoch {epoch}/{max_epochs} - "
                  f"Loss: {total_loss.item():.6f} "
                  f"(Data: {loss_L.item():.6f}, Physics: {loss_physics.item():.6f}), "
                  f"λ={current_lambda:.4f}, {weight_info}, "
                  f"R²: {train_r2:.4f}")
    
    print(f"\n✅ Decoupled prediction training complete!")
    save_physics_weight_history(history, output_folders)
    
    return model, history

def evaluate_decoupled_model_at_checkpoint(model, processor, X_test, L_test, direction_test, case_test_ids, output_folders):
    """Checkpoint evaluation for decoupled prediction model"""
    model.eval()
    device = next(model.parameters()).device
    
    # Convert test data
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        # Predict L value
        L_pred = model(X_test_tensor).cpu().numpy().flatten()
    
    # Handle different formats of L_test
    if isinstance(L_test, pd.Series):
        L_test_array = L_test.values
    elif isinstance(L_test, np.ndarray):
        L_test_array = L_test
    else:
        L_test_array = np.array(L_test)
    
    # Transform back to original scale
    if processor.use_L_normalization:
        L_pred_original = processor.inverse_transform_L(L_pred)
        L_test_original = processor.inverse_transform_L(L_test_array)
    else:
        L_pred_original = L_pred
        L_test_original = L_test_array
    
    # Calculate evaluation metrics
    from sklearn.metrics import mean_squared_error, r2_score
    
    mse = mean_squared_error(L_test_original, L_pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(L_test_original, L_pred_original)
    
    # Calculate relative error
    relative_error = np.abs(L_pred_original - L_test_original) / (L_test_original + 1e-8) * 100
    
    print(f"  L value prediction performance:")
    print(f"    RMSE: {rmse:.4f} MPa")
    print(f"    R²: {r2:.4f}")
    print(f"    Mean relative error: {relative_error.mean():.2f}%")
    print(f"    L value prediction range: [{L_pred_original.min():.2f}, {L_pred_original.max():.2f}] MPa")
    print(f"    L value true range: [{L_test_original.min():.2f}, {L_test_original.max():.2f}] MPa")
    
    # Reconstruct stress tensor for additional analysis
    stress_reconstructed = processor.reconstruct_stress_tensor(L_pred_original, direction_test)
    print(f"    Reconstructed stress range: [{stress_reconstructed.min():.2f}, {stress_reconstructed.max():.2f}] MPa")

def save_physics_weight_history(history, output_dir):
    """Save detailed record of physics loss weight changes"""
    from utils import get_timestamped_filename
    import os
    
    report_filename = get_timestamped_filename('physics_weight_history', 'txt')
    report_path = os.path.join(output_dir, 'reports', report_filename)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Fix: use correct key names to get data from history dictionary
    weights = np.array(history['physics_weight'])
    adaptive_weights = np.array(history['adaptive_weight'])
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Physics Loss Weight Change Detailed Record (Adaptive Dynamic Strategy)\n")
        f.write("="*80 + "\n\n")
        
        f.write("Training strategy description:\n")
        f.write("  Phase 1 (Warmup period): Physics weight = 0, focus on data fitting\n")
        f.write("  Phase 2 (Progressive period): Physics weight gradually increases\n")
        f.write("  Phase 3 (Adaptive period): Dynamically adjust weight based on gradients\n\n")
        
        f.write("Training statistics:\n")
        f.write(f"  Total training epochs: {len(history['epoch'])}\n")
        f.write(f"  Initial effective weight: {weights[0]:.6f}\n")
        f.write(f"  Final effective weight: {weights[-1]:.6f}\n")
        f.write(f"  Maximum effective weight: {weights.max():.6f}\n")
        f.write(f"  Average effective weight: {weights.mean():.6f}\n\n")
        
        zero_weights = weights[weights == 0]
        nonzero_weights = weights[weights > 0]
        
        f.write("Weight analysis:\n")
        f.write(f"  Warmup phase (weight=0) epochs: {len(zero_weights)}\n")
        if len(nonzero_weights) > 0:
            f.write(f"  Active phase epochs: {len(nonzero_weights)}\n")
            f.write(f"  Active phase average weight: {nonzero_weights.mean():.6f}\n")
            f.write(f"  Weight standard deviation: {nonzero_weights.std():.6f}\n")
            f.write(f"  Adaptive weight range: [{adaptive_weights[adaptive_weights>0].min():.6f}, {adaptive_weights.max():.6f}]\n")
    
    print(f"✅ Physics weight change history saved: {report_path}")

# Maintain backward compatible interface
def train_tsai_wu_pinn_stable(*args, **kwargs):
    """Backward compatible interface"""
    return train_decoupled_tsai_wu_pinn(*args, **kwargs)
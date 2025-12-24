# trainer.py
# Responsibility: Core training and validation logic for the model.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd

# Import necessary classes and functions from other modules
from model import TsaiWuPINN
from physics_loss import TsaiWuPhysicsLoss
from config import TSAI_WU_COEFFICIENTS

def train_decoupled_tsai_wu_pinn(X_train, L_train, case_train_ids, direction_train,
                                X_val, L_val, case_val_ids, direction_val,
                                processor, config, X_test=None, L_test=None, 
                                case_test_ids=None, direction_test=None, output_folders=None,
                                max_epochs=3000, early_stop_patience=500, verbose=True):
    """Decoupled prediction PINN training - flexible configuration supporting hyperparameter search"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"\n🚀 Decoupled prediction training mode started")
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
    W_physics = config['W_physics']
    
    # Use passed parameters or defaults
    total_epochs = max_epochs
    checkpoint_epoch = max(500, max_epochs // 6) if output_folders else max_epochs + 1  # Disable checkpoint output
    
    if verbose:
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
    if verbose:
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
    physics_calculator = TsaiWuPhysicsLoss(TSAI_WU_COEFFICIENTS, device, processor)
    
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
        raise ValueError(f"Data dimension mismatch: X({X_train_tensor.shape[0]}) vs L({L_train_tensor.shape[0]}) vs Direction({direction_train_tensor.shape[0]})")
    
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
    
    # Training history record
    history = {
        'train_losses': [], 'val_losses': [],
        'L_losses': [], 'val_L_losses': [],
        'physics_losses': [], 'lambda_values': [],
        'learning_rates': [], 'L_monitor_losses': []
    }
    
    print(f"\n🎯 Decoupled prediction training strategy:")
    print(f"📊 Phase 1 (Pure L value fitting): 1-{warm_start_epochs} epochs")
    print(f"🔧 Phase 2 (Progressive physics constraints): {warm_start_epochs+1}-{physics_delay_epochs} epochs")
    print(f"⚖️  Phase 3 (L value + physics joint optimization): {physics_delay_epochs+1}-{total_epochs} epochs")
    print("="*70)

    # Early stopping mechanism
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = early_stop_patience

    for epoch in range(total_epochs):
        epoch_num = epoch + 1
        
        # Lambda scheduling
        if epoch_num <= warm_start_epochs:
            lambda_t = 0.0
            stage_name = "Pure L fitting"
            stage_emoji = "📊"
        elif epoch_num <= physics_delay_epochs:
            progress = (epoch_num - warm_start_epochs) / (physics_delay_epochs - warm_start_epochs)
            lambda_t = lambda_final * progress
            stage_name = "Progressive physics"
            stage_emoji = "🔧"
        else:
            lambda_t = lambda_final
            stage_name = "L + physics joint"
            stage_emoji = "⚖️"

        # === Core decoupled prediction training step ===
        model.train()
        optimizer.zero_grad()
        
        # 1. Prediction step: input 20D features, output 1D failure length L
        L_pred = model(X_train_tensor)  # (batch_size, 1)
        
        # 2. Reconstruction step: L_pred * direction_vectors -> 6D stress tensor
        stress_reconstructed = L_pred * direction_train_tensor  # (batch_size, 6)
        
        # 3. Loss calculation
        # 3a. Data loss: MSE of L values (main loss)
        loss_L = nn.MSELoss()(L_pred, L_train_tensor)
        
        # 3b. Physics loss: violation degree of Tsai-Wu criterion on reconstructed stress tensor
        if lambda_t > 0 and W_physics > 0:
            # Reconstructed stress tensor is already in original scale, pass directly to physics loss calculator
            dummy_true_stress = torch.zeros_like(stress_reconstructed)
            try:
                _, _, loss_physics, loss_L_monitor = physics_calculator.compute_all_normalized_loss(
                    stress_reconstructed, dummy_true_stress,
                    case_train_tensor, lambda_t, W_physics
                )
            except Exception as e:
                print(f"⚠️ Physics loss calculation error: {e}")
                loss_physics = torch.tensor(0.0, device=device)
                loss_L_monitor = torch.tensor(0.0, device=device)
        else:
            loss_physics = torch.tensor(0.0, device=device)
            loss_L_monitor = torch.tensor(0.0, device=device)
        
        # 3c. Total loss - dynamically adjust physics loss weight
        physics_weight_factor = 0.1 if epoch_num <= warm_start_epochs else 0.2  # Stage-wise adjustment
        total_loss = loss_L + lambda_t * W_physics * physics_weight_factor * loss_physics
        
        # Backpropagation - enhanced numerical stability
        if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() > 1e6:
            print(f"⚠️ Epoch {epoch_num}: Abnormal loss ({total_loss.item():.2e}), skipping backpropagation")
            continue
        
        total_loss.backward()
        
        # Gradient clipping and checking
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if grad_norm > 10.0:
            print(f"⚠️ Epoch {epoch_num}: Gradient norm too large ({grad_norm:.2f}), clipped")
        
        optimizer.step()
        
        # === Validation step ===
        model.eval()
        with torch.no_grad():
            if len(X_val) > 0:
                # Predict validation set L values
                L_val_pred = model(X_val_tensor)
                
                # Reconstruct validation set stress tensor
                stress_val_reconstructed = L_val_pred * direction_val_tensor
                
                # Calculate validation loss
                val_loss_L = nn.MSELoss()(L_val_pred, L_val_tensor)
                
                if lambda_t > 0 and W_physics > 0:
                    dummy_val_true_stress = torch.zeros_like(stress_val_reconstructed)
                    try:
                        _, _, val_loss_physics, val_loss_L_monitor = physics_calculator.compute_all_normalized_loss(
                            stress_val_reconstructed, dummy_val_true_stress,
                            case_val_tensor, lambda_t, W_physics
                        )
                    except:
                        val_loss_physics = torch.tensor(0.0, device=device)
                        val_loss_L_monitor = torch.tensor(0.0, device=device)
                else:
                    val_loss_physics = torch.tensor(0.0, device=device)
                    val_loss_L_monitor = torch.tensor(0.0, device=device)
                
                val_total_loss = val_loss_L + lambda_t * W_physics * 0.1 * val_loss_physics
            else:
                # Use training loss when no validation set
                val_loss_L = loss_L
                val_loss_physics = loss_physics
                val_total_loss = total_loss
                val_loss_L_monitor = loss_L_monitor
        
        # Learning rate scheduling
        scheduler.step(val_total_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Early stopping check
        if val_total_loss < best_val_loss * 0.995:  # Allow 0.5% fluctuation
            best_val_loss = val_total_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Dynamically adjust early stopping condition
        current_patience_limit = max_patience
        if epoch_num <= warm_start_epochs:
            current_patience_limit = max_patience * 2  # More lenient in early stage
        
        if patience_counter > current_patience_limit and epoch_num > warm_start_epochs:
            print(f"🛑 Early stopping: validation loss not improved for {patience_counter} epochs (threshold: {current_patience_limit})")
            break
        
        # Record history
        history['train_losses'].append(total_loss.item())
        history['val_losses'].append(val_total_loss.item())
        history['L_losses'].append(loss_L.item())
        history['val_L_losses'].append(val_loss_L.item())
        history['physics_losses'].append(loss_physics.item())
        history['lambda_values'].append(lambda_t)
        history['learning_rates'].append(current_lr)
        history['L_monitor_losses'].append(loss_L_monitor.item())
        
        # Print progress - control output based on verbose parameter
        print_interval = 100 if verbose else 1000
        if verbose and (epoch_num % print_interval == 0 or epoch_num <= 10):
            print(f"{stage_emoji} Epoch {epoch_num:4d}/{total_epochs} | {stage_name}")
            print(f"   Total loss: {total_loss.item():.6f} | L loss: {loss_L.item():.6f} | Physics loss: {loss_physics.item():.6f}")
            print(f"   Val L loss: {val_loss_L.item():.6f} | λ: {lambda_t:.3f} | Physics weight: {physics_weight_factor:.1f}")
            print(f"   L prediction range: [{L_pred.min().item():.4f}, {L_pred.max().item():.4f}] | LR: {current_lr:.2e}")
            print(f"   Gradient norm: {grad_norm:.4f} | Early stop counter: {patience_counter}/{current_patience_limit}")
            
            L_pred_std = L_pred.std().item()
            if L_pred_std < 1e-4:
                print(f"   ⚠️ Warning: L prediction std too small ({L_pred_std:.2e}), model may be stuck in local optimum")
        
        # Checkpoint evaluation
        if output_folders and epoch_num % checkpoint_epoch == 0 and X_test is not None:
            if verbose:
                print(f"\n📊 Epoch {epoch_num} decoupled prediction checkpoint evaluation:")
            evaluate_decoupled_model_at_checkpoint(
                model, processor, X_test, L_test, direction_test, case_test_ids, output_folders
            )
    
    if verbose:
        print(f"\n✅ Decoupled prediction training complete!")
    return model, history

def evaluate_decoupled_model_at_checkpoint(model, processor, X_test, L_test, direction_test, case_test_ids, output_folders):
    """Checkpoint evaluation for decoupled prediction model"""
    model.eval()
    device = next(model.parameters()).device
    
    # Convert test data
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        # Predict L values
        L_pred = model(X_test_tensor).cpu().numpy().flatten()
    
    # Handle different formats of L_test
    if isinstance(L_test, pd.Series):
        L_test_array = L_test.values
    elif isinstance(L_test, np.ndarray):
        L_test_array = L_test
    else:
        L_test_array = np.array(L_test)
    
    # Convert back to original scale
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
    
    # Reconstruct stress tensor for additional analysis
    stress_reconstructed = processor.reconstruct_stress_tensor(L_pred_original, direction_test)
    
    print(f"Checkpoint evaluation results:")
    print(f"  L value prediction performance:")
    print(f"    RMSE: {rmse:.4f} MPa")
    print(f"    R²: {r2:.4f}")
    print(f"    Average relative error: {relative_error.mean():.2f}%")
    print(f"    L value prediction range: [{L_pred_original.min():.2f}, {L_pred_original.max():.2f}] MPa")
    print(f"    L value true range: [{L_test_original.min():.2f}, {L_test_original.max():.2f}] MPa")
    print(f"    Reconstructed stress range: [{stress_reconstructed.min():.2f}, {stress_reconstructed.max():.2f}] MPa")

# Backward compatibility interface
def train_tsai_wu_pinn_stable(*args, **kwargs):
    """Backward compatibility interface"""
    return train_decoupled_tsai_wu_pinn(*args, **kwargs)
# trainer.py
# Responsibility: Contains core logic for model training and validation.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd

from model import DualConstraintPINN
from dual_physics_loss import DualPhysicsLoss
from config import TSAI_WU_COEFFICIENTS, CUNTZE_PARAMETERS

def train_dual_constraint_pinn(X_train, L_train, case_train_ids, direction_train,
                              X_val, L_val, case_val_ids, direction_val,
                              processor, config, X_test=None, L_test=None, 
                              case_test_ids=None, direction_test=None, output_folders=None,
                              max_epochs=3000, early_stop_patience=500, verbose=True):
    """Dual constraint PINN training - Tsai-Wu + Cuntze"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"\n🚀 Dual constraint PINN training started")
        print(f"🔧 Physics constraints: Tsai-Wu + Cuntze dual criteria")
        print(f"💻 Training device: {device}")
    
    # Extract configuration parameters
    layer_sizes = config['layer_sizes']
    activation = config['activation']
    lr = config['lr']
    dropout_rate = config['dropout_rate']
    weight_decay = config['weight_decay']
    
    # Dual constraint parameters
    lambda_tsai_wu_final = config.get('lambda_tsai_wu_final', 0.3)
    lambda_cuntze_final = config.get('lambda_cuntze_final', 0.3)
    tsai_wu_delay_epochs = config.get('tsai_wu_delay_epochs', 1500)
    cuntze_delay_epochs = config.get('cuntze_delay_epochs', 1600)
    warm_start_epochs = config.get('warm_start_epochs', 800)
    
    W_tsai_wu = config.get('W_tsai_wu', 0.15)
    W_cuntze = config.get('W_cuntze', 0.15)
    constraint_balance_alpha = config.get('constraint_balance_alpha', 0.5)
    
    # SA-PINN adaptive weight parameters
    use_sa_pinn = config.get('use_sa_pinn', False)
    sa_pinn_lr = config.get('sa_pinn_lr', 0.001)
    weight_update_freq = config.get('weight_update_freq', 10)
    
    print(f"\n📊 Dual constraint configuration:")
    print(f"   Tsai-Wu: λ={lambda_tsai_wu_final}, W={W_tsai_wu}, delay={tsai_wu_delay_epochs}")
    print(f"   Cuntze: λ={lambda_cuntze_final}, W={W_cuntze}, delay={cuntze_delay_epochs}")
    print(f"   Constraint balance α={constraint_balance_alpha}")
    print(f"   SA-PINN adaptive weighting: {'Enabled' if use_sa_pinn else 'Disabled'}")
    
    # Create model
    model = DualConstraintPINN(
        layer_sizes=layer_sizes,
        dropout_rate=dropout_rate,
        activation=activation
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=100, min_lr=1e-7)
    
    # Dual physics loss calculator
    physics_calculator = DualPhysicsLoss(TSAI_WU_COEFFICIENTS, CUNTZE_PARAMETERS, device, processor)
    
    # Data conversion
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    L_train_tensor = torch.FloatTensor(L_train).to(device)
    if L_train_tensor.ndim == 1:
        L_train_tensor = L_train_tensor.unsqueeze(1)
    
    direction_train_tensor = torch.FloatTensor(direction_train).to(device)
    case_train_tensor = torch.FloatTensor(case_train_ids.values if hasattr(case_train_ids, 'values') else case_train_ids).to(device)
    
    # Validation set data
    if len(X_val) > 0:
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        L_val_tensor = torch.FloatTensor(L_val).to(device)
        if L_val_tensor.ndim == 1:
            L_val_tensor = L_val_tensor.unsqueeze(1)
        direction_val_tensor = torch.FloatTensor(direction_val).to(device)
        case_val_tensor = torch.FloatTensor(case_val_ids.values if hasattr(case_val_ids, 'values') else case_val_ids).to(device)
    else:
        X_val_tensor = torch.empty(0, X_train.shape[1]).to(device)
        L_val_tensor = torch.empty(0, 1).to(device)
        direction_val_tensor = torch.empty(0, 6).to(device)
        case_val_tensor = torch.empty(0).to(device)
    
    # Training history
    history = {
        'train_losses': [], 'val_losses': [],
        'L_losses': [], 'val_L_losses': [],
        'tsai_wu_losses': [], 'cuntze_losses': [],
        'lambda_tsai_wu_values': [], 'lambda_cuntze_values': [],
        'learning_rates': [], 'constraint_balance': []
    }
    
    print(f"\n🎯 Dual constraint training strategy:")
    print(f"📊 Phase 1 (Pure L value fitting): 1-{warm_start_epochs}")
    print(f"🔧 Phase 2 (Tsai-Wu introduction): {warm_start_epochs+1}-{tsai_wu_delay_epochs}")
    print(f"⚖️  Phase 3 (Dual constraints): {tsai_wu_delay_epochs+1}-{max_epochs}")
    print("="*70)
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(max_epochs):
        epoch_num = epoch + 1
        
        # Lambda scheduling strategy
        if epoch_num <= warm_start_epochs:
            lambda_tsai_wu = 0.0
            lambda_cuntze = 0.0
            stage_name = "Pure L value fitting"
            stage_emoji = "📊"
        elif epoch_num <= tsai_wu_delay_epochs:
            progress = (epoch_num - warm_start_epochs) / (tsai_wu_delay_epochs - warm_start_epochs)
            lambda_tsai_wu = lambda_tsai_wu_final * progress
            lambda_cuntze = 0.0
            stage_name = "Tsai-Wu constraint"
            stage_emoji = "🔧"
        elif epoch_num <= cuntze_delay_epochs:
            lambda_tsai_wu = lambda_tsai_wu_final
            progress = (epoch_num - tsai_wu_delay_epochs) / (cuntze_delay_epochs - tsai_wu_delay_epochs)
            lambda_cuntze = lambda_cuntze_final * progress
            stage_name = "Dual constraint transition"
            stage_emoji = "⚖️"
        else:
            lambda_tsai_wu = lambda_tsai_wu_final
            lambda_cuntze = lambda_cuntze_final
            stage_name = "Dual constraint"
            stage_emoji = "⚖️"
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        # Predict L value and reconstruct stress
        L_pred = model(X_train_tensor)
        stress_reconstructed = L_pred * direction_train_tensor
        
        # L value loss
        loss_L = nn.MSELoss()(L_pred, L_train_tensor)
        
        # Dual physics loss
        if (lambda_tsai_wu > 0 or lambda_cuntze > 0):
            dummy_true_stress = torch.zeros_like(stress_reconstructed)
            total_loss, loss_data, loss_tsai_wu, loss_cuntze = physics_calculator.compute_dual_physics_loss(
                stress_reconstructed, dummy_true_stress, case_train_tensor,
                lambda_tsai_wu * W_tsai_wu, lambda_cuntze * W_cuntze, 
                alpha=constraint_balance_alpha
            )
            
            # SA-PINN adaptive weight update
            if use_sa_pinn and epoch_num % weight_update_freq == 0:
                physics_calculator.update_adaptive_weights(loss_tsai_wu, loss_cuntze, sa_pinn_lr)
            
            # Total loss includes L loss and physics loss
            total_loss = loss_L + lambda_tsai_wu * W_tsai_wu * loss_tsai_wu + lambda_cuntze * W_cuntze * loss_cuntze
        else:
            total_loss = loss_L
            loss_tsai_wu = torch.tensor(0.0, device=device)
            loss_cuntze = torch.tensor(0.0, device=device)
        
        # Backward propagation
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"⚠️ Epoch {epoch_num}: Loss abnormal, skipping")
            continue
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation step
        model.eval()
        with torch.no_grad():
            if len(X_val) > 0:
                L_val_pred = model(X_val_tensor)
                stress_val_reconstructed = L_val_pred * direction_val_tensor
                val_loss_L = nn.MSELoss()(L_val_pred, L_val_tensor)
                
                if lambda_tsai_wu > 0 or lambda_cuntze > 0:
                    dummy_val_true_stress = torch.zeros_like(stress_val_reconstructed)
                    val_total_loss, _, val_loss_tsai_wu, val_loss_cuntze = physics_calculator.compute_dual_physics_loss(
                        stress_val_reconstructed, dummy_val_true_stress, case_val_tensor,
                        lambda_tsai_wu * W_tsai_wu, lambda_cuntze * W_cuntze,
                        alpha=constraint_balance_alpha
                    )
                    val_total_loss = val_loss_L + lambda_tsai_wu * W_tsai_wu * val_loss_tsai_wu + lambda_cuntze * W_cuntze * val_loss_cuntze
                else:
                    val_total_loss = val_loss_L
            else:
                val_loss_L = loss_L
                val_total_loss = total_loss
        
        # Learning rate scheduling
        scheduler.step(val_total_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Early stopping
        if val_total_loss < best_val_loss * 0.995:
            best_val_loss = val_total_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter > early_stop_patience and epoch_num > warm_start_epochs:
            print(f"🛑 Early stopping: validation loss not improved for {patience_counter} epochs")
            break
        
        # Record history
        history['train_losses'].append(total_loss.item())
        history['val_losses'].append(val_total_loss.item())
        history['L_losses'].append(loss_L.item())
        history['val_L_losses'].append(val_loss_L.item())
        history['tsai_wu_losses'].append(loss_tsai_wu.item())
        history['cuntze_losses'].append(loss_cuntze.item())
        history['lambda_tsai_wu_values'].append(lambda_tsai_wu)
        history['lambda_cuntze_values'].append(lambda_cuntze)
        history['learning_rates'].append(current_lr)
        history['constraint_balance'].append(constraint_balance_alpha)
        
        # Print progress
        if verbose and (epoch_num % 100 == 0 or epoch_num <= 10):
            print(f"{stage_emoji} Epoch {epoch_num:4d}/{max_epochs} | {stage_name}")
            print(f"   Total loss: {total_loss.item():.6f} | L loss: {loss_L.item():.6f}")
            print(f"   Tsai-Wu: {loss_tsai_wu.item():.6f} (λ={lambda_tsai_wu:.3f})")
            print(f"   Cuntze: {loss_cuntze.item():.6f} (λ={lambda_cuntze:.3f})")
            print(f"   Validation loss: {val_total_loss.item():.6f} | LR: {current_lr:.2e}")
    
    if verbose:
        print(f"\n✅ Dual constraint PINN training complete!")
    
    return model, history

# Backward compatibility interface
def train_decoupled_tsai_wu_pinn(*args, **kwargs):
    """Backward compatible training interface"""
    return train_dual_constraint_pinn(*args, **kwargs)

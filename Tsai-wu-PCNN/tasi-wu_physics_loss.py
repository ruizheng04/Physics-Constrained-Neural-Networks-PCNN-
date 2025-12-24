# physics_loss.py
# Responsibility: Encapsulate and compute physics loss based on Tsai-Wu criterion.

import torch
import torch.nn.functional as F
import numpy as np

class TsaiWuPhysicsLoss:
    """Redesigned Tsai-Wu criterion physics loss calculator - supports decoupled prediction mode"""
    
    def __init__(self, tsai_wu_coeffs, device, processor=None):
        self.device = device
        self.processor = processor
        
        # Keep original coefficients (only for validation and visualization)
        self.original_coeffs = {}
        for case_id, coeffs in tsai_wu_coeffs.items():
            self.original_coeffs[case_id] = {
                k: torch.tensor(v, dtype=torch.float32, device=device) 
                for k, v in coeffs.items()
            }
        
        # Transformation coefficients for normalized space
        self.normalized_coeffs = {}
        self.use_normalized_coeffs = False
        
        # Check if in decoupled prediction mode and create virtual stress normalization parameters
        if processor and processor.is_fitted:
            if hasattr(processor, 'use_L_normalization') and processor.use_L_normalization:
                if hasattr(processor, 'original_stress_tensor'):
                    print("🔧 Decoupled prediction mode: creating normalization parameters from original stress data...")
                    self._create_stress_normalization_from_original_data(processor)
                else:
                    print("⚠️ Decoupled prediction mode: cannot find original stress data, using original coefficients")
            elif hasattr(processor, 'use_stress_normalization') and processor.use_stress_normalization:
                self._load_existing_stress_normalization(processor)
        else:
            print("⚠️ Normalization not enabled or processor not fitted, using original coefficients")

    def _create_stress_normalization_from_original_data(self, processor):
        """Create normalization parameters from original stress data in decoupled prediction mode"""
        try:
            original_stress = processor.original_stress_tensor
            
            # Calculate min and max for each column
            stress_min = np.min(original_stress, axis=0)
            stress_max = np.max(original_stress, axis=0)
            stress_range = stress_max - stress_min
            
            # Handle zero range cases
            for i, range_val in enumerate(stress_range):
                if range_val < 1e-6:
                    stress_range[i] = 1.0  # Set default range
                    print(f"⚠️ Stress component {i} has minimal range, setting to default value")
            
            # Create normalization parameters
            self.stress_min = torch.tensor(stress_min, dtype=torch.float32, device=self.device)
            self.stress_range = torch.tensor(stress_range, dtype=torch.float32, device=self.device)
            
            # Transform Tsai-Wu coefficients to normalized space
            self.normalized_coeffs = self._transform_coefficients_to_normalized_space(self.original_coeffs)
            self.use_normalized_coeffs = True
            
            print("✅ Decoupled prediction mode: successfully created stress normalization parameters")
            print(f"📊 Stress range: min={stress_min}, max={stress_max}")
            
        except Exception as e:
            print(f"❌ Failed to create stress normalization parameters: {e}")
            self.use_normalized_coeffs = False

    def _load_existing_stress_normalization(self, processor):
        """Load existing stress normalization parameters for traditional mode"""
        try:
            if hasattr(processor, 'stress_scaler') and hasattr(processor.stress_scaler, 'data_min_'):
                self.stress_min = torch.tensor(processor.stress_scaler.data_min_, dtype=torch.float32, device=self.device)
                self.stress_range = torch.tensor(processor.stress_scaler.data_range_, dtype=torch.float32, device=self.device)
                self.normalized_coeffs = self._transform_coefficients_to_normalized_space(self.original_coeffs)
                self.use_normalized_coeffs = True
                print("✅ Traditional mode: loaded existing stress normalization parameters")
            else:
                print("⚠️ Traditional mode: valid stress normalization parameters not found")
                self.use_normalized_coeffs = False
        except Exception as e:
            print(f"❌ Failed to load stress normalization parameters: {e}")
            self.use_normalized_coeffs = False

    def _transform_coefficients_to_normalized_space(self, original_coeffs):
        """Transform Tsai-Wu coefficients to normalized space"""
        transformed = {}
        
        print("🔧 Starting Tsai-Wu coefficient transformation to normalized space...")
        
        for case_id, coeffs in original_coeffs.items():
            # Extract original coefficients
            F1, F2, F3 = coeffs['F1'], coeffs['F2'], coeffs['F3']
            F11, F22, F33 = coeffs['F11'], coeffs['F22'], coeffs['F33']
            F44, F55, F66 = coeffs['F44'], coeffs['F55'], coeffs['F66']
            F12, F13, F23 = coeffs['F12'], coeffs['F13'], coeffs['F23']
            
            # Normalization parameters
            r = self.stress_range  # [range_x, range_y, range_z, range_xy, range_yz, range_xz]
            m = self.stress_min    # [min_x, min_y, min_z, min_xy, min_yz, min_xz]
            
            # New quadratic term coefficients
            F11_norm = F11 * r[0] * r[0]
            F22_norm = F22 * r[1] * r[1]
            F33_norm = F33 * r[2] * r[2]
            F44_norm = F44 * r[3] * r[3]
            F55_norm = F55 * r[4] * r[4]
            F66_norm = F66 * r[5] * r[5]
            
            # Cross-term coefficients
            F12_norm = F12 * r[0] * r[1]
            F13_norm = F13 * r[0] * r[2]
            F23_norm = F23 * r[1] * r[2]
            
            # New linear term coefficients
            F1_norm = (F1 * r[0] + 
                      2 * F11 * r[0] * m[0] + 
                      2 * F12 * r[0] * m[1] + 
                      2 * F13 * r[0] * m[2])
            
            F2_norm = (F2 * r[1] +
                      2 * F22 * r[1] * m[1] +
                      2 * F12 * r[1] * m[0] +
                      2 * F23 * r[1] * m[2])
            
            F3_norm = (F3 * r[2] +
                      2 * F33 * r[2] * m[2] +
                      2 * F13 * r[2] * m[0] +
                      2 * F23 * r[2] * m[1])
            
            # Shear stress linear terms
            F4_norm = F44 * 2 * r[3] * m[3]
            F5_norm = F55 * 2 * r[4] * m[4]
            F6_norm = F66 * 2 * r[5] * m[5]
            
            # Constant term
            constant_term = (F1 * m[0] + F2 * m[1] + F3 * m[2] +
                           F11 * m[0]**2 + F22 * m[1]**2 + F33 * m[2]**2 +
                           F44 * m[3]**2 + F55 * m[4]**2 + F66 * m[5]**2 +
                           2 * F12 * m[0] * m[1] + 2 * F13 * m[0] * m[2] + 2 * F23 * m[1] * m[2])
            
            target_value = 1.0 - constant_term
            
            transformed[case_id] = {
                'F1_norm': torch.tensor(float(F1_norm), dtype=torch.float32, device=self.device),
                'F2_norm': torch.tensor(float(F2_norm), dtype=torch.float32, device=self.device),
                'F3_norm': torch.tensor(float(F3_norm), dtype=torch.float32, device=self.device),
                'F4_norm': torch.tensor(float(F4_norm), dtype=torch.float32, device=self.device),
                'F5_norm': torch.tensor(float(F5_norm), dtype=torch.float32, device=self.device),
                'F6_norm': torch.tensor(float(F6_norm), dtype=torch.float32, device=self.device),
                'F11_norm': torch.tensor(float(F11_norm), dtype=torch.float32, device=self.device),
                'F22_norm': torch.tensor(float(F22_norm), dtype=torch.float32, device=self.device),
                'F33_norm': torch.tensor(float(F33_norm), dtype=torch.float32, device=self.device),
                'F44_norm': torch.tensor(float(F44_norm), dtype=torch.float32, device=self.device),
                'F55_norm': torch.tensor(float(F55_norm), dtype=torch.float32, device=self.device),
                'F66_norm': torch.tensor(float(F66_norm), dtype=torch.float32, device=self.device),
                'F12_norm': torch.tensor(float(F12_norm), dtype=torch.float32, device=self.device),
                'F13_norm': torch.tensor(float(F13_norm), dtype=torch.float32, device=self.device),
                'F23_norm': torch.tensor(float(F23_norm), dtype=torch.float32, device=self.device),
                'target': torch.tensor(float(target_value), dtype=torch.float32, device=self.device)
            }
            
            if case_id == 1:
                print(f"  Case {case_id} transformation example:")
                print(f"    F1: {F1:.2e} -> {F1_norm:.2e}")
                print(f"    F11: {F11:.2e} -> {F11_norm:.2e}")
                print(f"    Target value: {target_value:.6f}")
        
        print("✅ Tsai-Wu coefficient transformation complete")
        return transformed

    def _compute_tsai_wu_loss_normalized(self, stress_normalized, case_ids_batch):
        """Compute Tsai-Wu criterion physics loss directly in normalized space"""
        batch_size = stress_normalized.shape[0]
        total_loss = torch.tensor(0.0, device=self.device)
        
        for i in range(batch_size):
            case_id = int(case_ids_batch[i].item())
            if case_id not in self.normalized_coeffs:
                case_id = 1  # Use default case
            
            coeffs = self.normalized_coeffs[case_id]
            s = stress_normalized[i]  # [σx_norm, σy_norm, σz_norm, τxy_norm, τyz_norm, τxz_norm]
            
            # Calculate Tsai-Wu criterion value in normalized space
            f_normalized = (
                # Linear terms
                coeffs['F1_norm'] * s[0] + coeffs['F2_norm'] * s[1] + coeffs['F3_norm'] * s[2] +
                coeffs['F4_norm'] * s[3] + coeffs['F5_norm'] * s[4] + coeffs['F6_norm'] * s[5] +
                
                # Quadratic terms
                coeffs['F11_norm'] * s[0]**2 + coeffs['F22_norm'] * s[1]**2 + coeffs['F33_norm'] * s[2]**2 +
                coeffs['F44_norm'] * s[3]**2 + coeffs['F55_norm'] * s[4]**2 + coeffs['F66_norm'] * s[5]**2 +
                
                # Cross terms
                2 * coeffs['F12_norm'] * s[0] * s[1] +
                2 * coeffs['F13_norm'] * s[0] * s[2] +
                2 * coeffs['F23_norm'] * s[1] * s[2]
            )
            
            # Calculate deviation from target value
            violation = (f_normalized - coeffs['target']) ** 2
            total_loss += violation
        
        return total_loss / batch_size

    def _compute_tsai_wu_loss_original(self, stress_original, case_ids_batch):
        """Compute Tsai-Wu criterion physics loss in original space (for comparison only)"""
        batch_size = stress_original.shape[0]
        total_loss = torch.tensor(0.0, device=self.device)
        
        for i in range(batch_size):
            case_id = int(case_ids_batch[i].item())
            if case_id not in self.original_coeffs:
                case_id = 1
            
            coeffs = self.original_coeffs[case_id]
            s = stress_original[i]
            
            f_value = (coeffs['F1'] * s[0] + coeffs['F2'] * s[1] + coeffs['F3'] * s[2] +
                      coeffs['F11'] * s[0]**2 + coeffs['F22'] * s[1]**2 + coeffs['F33'] * s[2]**2 +
                      coeffs['F44'] * s[3]**2 + coeffs['F55'] * s[4]**2 + coeffs['F66'] * s[5]**2 +
                      2 * coeffs['F12'] * s[0] * s[1] +
                      2 * coeffs['F13'] * s[0] * s[2] +
                      2 * coeffs['F23'] * s[1] * s[2])
            
            violation = (f_value - 1.0) ** 2
            total_loss += violation
        
        return total_loss / batch_size

    def _compute_L_monitor_normalized(self, stress_pred_normalized, stress_true_normalized):
        """Compute failure length monitoring loss in normalized space"""
        L_pred = torch.sqrt(torch.sum(stress_pred_normalized**2, dim=1))
        L_true = torch.sqrt(torch.sum(stress_true_normalized**2, dim=1))
        return F.mse_loss(L_pred, L_true)

    def compute_all_normalized_loss(self, stress_pred_normalized, stress_true_normalized, 
                                   case_ids_batch, lambda_weight, W_physics):
        """Compute all losses in normalized space - supports decoupled prediction mode"""
        
        # 1. Data loss - for decoupled prediction, this is reconstructed stress loss
        loss_data_stress = F.mse_loss(stress_pred_normalized, stress_true_normalized)
        
        # 2. Physics loss - using reconstructed stress tensor
        if lambda_weight > 0 and W_physics > 0:
            if self.use_normalized_coeffs:
                # Need to convert stress tensor from original scale to normalized scale
                stress_for_physics = self._convert_to_normalized_space(stress_pred_normalized)
                loss_physics = self._compute_tsai_wu_loss_normalized(stress_for_physics, case_ids_batch)
            else:
                # If no normalization coefficients, compute in original space
                loss_physics = self._compute_tsai_wu_loss_original(stress_pred_normalized, case_ids_batch)
        else:
            loss_physics = torch.tensor(0.0, device=self.device)
        
        # 3. Total loss
        total_loss = loss_data_stress + lambda_weight * W_physics * loss_physics
        
        # 4. Monitoring metrics
        loss_L_monitor = self._compute_L_monitor_normalized(stress_pred_normalized, stress_true_normalized)
        
        return total_loss, loss_data_stress, loss_physics, loss_L_monitor

    def _convert_to_normalized_space(self, stress_original):
        """Convert stress from original scale to normalized space"""
        if self.use_normalized_coeffs:
            # Apply Min-Max normalization: (stress - min) / range
            return (stress_original - self.stress_min.unsqueeze(0)) / self.stress_range.unsqueeze(0)
        else:
            return stress_original

    def compute_adaptive_normalized_loss(self, stress_pred_normalized, stress_true_normalized, 
                                        case_ids_batch, lambda_weight, alpha=0.1):
        """Compute adaptive weight loss in normalized space"""
        
        # 1. Data loss - in normalized space
        loss_data_stress = F.mse_loss(stress_pred_normalized, stress_true_normalized)
        
        # 2. Physics loss - in normalized space
        if lambda_weight > 0:
            if self.use_normalized_coeffs:
                loss_physics = self._compute_tsai_wu_loss_normalized(stress_pred_normalized, case_ids_batch)
            else:
                loss_physics = torch.tensor(0.0, device=self.device)
        else:
            loss_physics = torch.tensor(0.0, device=self.device)
            adaptive_weight = torch.tensor(1.0, device=self.device)
            total_loss = loss_data_stress
            loss_L_monitor = self._compute_L_monitor_normalized(stress_pred_normalized, stress_true_normalized)
            return total_loss, loss_data_stress, loss_physics, adaptive_weight, loss_L_monitor
        
        # 3. Adaptive weight calculation (simplified version)
        current_data_loss = loss_data_stress.item()
        current_physics_loss = loss_physics.item()
        
        if current_physics_loss > 1e-8:
            # Target: physics loss contribution is alpha times data loss
            target_contribution = alpha * current_data_loss
            adaptive_weight = torch.tensor(target_contribution / (lambda_weight * current_physics_loss), device=self.device)
            adaptive_weight = torch.clamp(adaptive_weight, 0.1, 5.0)
        else:
            adaptive_weight = torch.tensor(1.0, device=self.device)
        
        # 4. Total loss
        total_loss = loss_data_stress + lambda_weight * adaptive_weight * loss_physics
        
        # 5. Monitoring metrics
        loss_L_monitor = self._compute_L_monitor_normalized(stress_pred_normalized, stress_true_normalized)
        
        return total_loss, loss_data_stress, loss_physics, adaptive_weight, loss_L_monitor

    # Backward compatibility interfaces
    def compute_fixed_weight_loss(self, stress_pred_scaled, stress_true_scaled, 
                                 case_ids_batch, lambda_weight, W_physics):
        """Backward compatibility interface - redirect to normalized loss computation"""
        return self.compute_all_normalized_loss(stress_pred_scaled, stress_true_scaled, 
                                               case_ids_batch, lambda_weight, W_physics)

    def compute_stable_adaptive_loss(self, stress_pred_scaled, stress_true_scaled, 
                                   case_ids_batch, lambda_weight, alpha=0.1):
        """Backward compatibility interface - redirect to normalized adaptive loss computation"""
        return self.compute_adaptive_normalized_loss(stress_pred_scaled, stress_true_scaled, 
                                                    case_ids_batch, lambda_weight, alpha)
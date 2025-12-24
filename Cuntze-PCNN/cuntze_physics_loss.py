# cuntze_physics_loss.py
# Responsibility: Implement Cuntze criterion physics loss calculation

import torch
import torch.nn.functional as F
import numpy as np

class CuntzePhysicsLoss:
    """Cuntze criterion physics loss calculator - fiber failure and matrix failure judgment"""
    
    def __init__(self, cuntze_params, device, processor=None):
        self.device = device
        self.processor = processor
        
        # Keep original Cuntze parameters
        self.original_params = {}
        for case_id, params in cuntze_params.items():
            self.original_params[case_id] = {
                k: torch.tensor(v, dtype=torch.float32, device=device) 
                for k, v in params.items()
            }
        
        # Transformation parameters in normalized space
        self.normalized_params = {}
        self.use_normalized_params = False
        
        # Check normalization settings
        if processor and processor.is_fitted:
            if hasattr(processor, 'use_L_normalization') and processor.use_L_normalization:
                # Decoupled prediction mode: create normalized parameters from original stress data
                if hasattr(processor, 'original_stress_tensor'):
                    print("🔧 Cuntze mode: Creating normalized parameters from original stress data...")
                    self._create_stress_normalization_from_original_data(processor)
                else:
                    print("⚠️ Cuntze mode: Cannot find original stress data, using original parameters")
            elif hasattr(processor, 'use_stress_normalization') and processor.use_stress_normalization:
                # Traditional mode: use existing stress normalization parameters
                self._load_existing_stress_normalization(processor)
        else:
            print("⚠️ Cuntze: Normalization not enabled or processor not fitted, using original parameters")

    def _create_stress_normalization_from_original_data(self, processor):
        """Create normalized parameters from original stress data in decoupled prediction mode"""
        try:
            original_stress = processor.original_stress_tensor
            
            # Calculate stress normalization parameters
            stress_min = np.min(original_stress, axis=0)
            stress_max = np.max(original_stress, axis=0)
            stress_range = stress_max - stress_min
            
            # Handle zero range
            for i, range_val in enumerate(stress_range):
                if range_val < 1e-6:
                    stress_range[i] = 1.0
                    print(f"⚠️ Cuntze: Stress component {i} range too small, set to default value")
            
            self.stress_min = torch.tensor(stress_min, dtype=torch.float32, device=self.device)
            self.stress_range = torch.tensor(stress_range, dtype=torch.float32, device=self.device)
            
            # Transform Cuntze parameters to normalized space
            self.normalized_params = self._transform_params_to_normalized_space(self.original_params)
            self.use_normalized_params = True
            
            print("✅ Cuntze: Successfully created stress normalization parameters")
            
        except Exception as e:
            print(f"❌ Cuntze: Failed to create normalized parameters: {e}")
            self.use_normalized_params = False

    def _load_existing_stress_normalization(self, processor):
        """Load existing stress normalization parameters in traditional mode"""
        try:
            if hasattr(processor, 'stress_scaler') and hasattr(processor.stress_scaler, 'data_min_'):
                self.stress_min = torch.tensor(processor.stress_scaler.data_min_, dtype=torch.float32, device=self.device)
                self.stress_range = torch.tensor(processor.stress_scaler.data_range_, dtype=torch.float32, device=self.device)
                self.normalized_params = self._transform_params_to_normalized_space(self.original_params)
                self.use_normalized_params = True
                print("✅ Cuntze: Loaded existing stress normalization parameters")
            else:
                print("⚠️ Cuntze: No valid stress normalization parameters found")
                self.use_normalized_params = False
        except Exception as e:
            print(f"❌ Cuntze: Failed to load normalized parameters: {e}")
            self.use_normalized_params = False

    def _transform_params_to_normalized_space(self, original_params):
        """Transform Cuntze parameters to normalized space"""
        transformed = {}
        
        print("🔧 Starting transformation of Cuntze parameters to normalized space...")
        
        for case_id, params in original_params.items():
            # Extract strength parameters and transform to normalized space
            # Stress strength needs to be scaled by dividing by the range
            r = self.stress_range  # [range_x, range_y, range_z, range_xy, range_yz, range_xz]
            
            # Fiber direction strength parameters (mainly related to σx)
            S_f_parallel_t_norm = params['S_f_parallel_t'] / r[0]
            S_f_parallel_c_norm = params['S_f_parallel_c'] / r[0]
            
            # Transverse strength parameters (related to σy, σz)
            S_f_perpendicular_norm_y = params['S_f_perpendicular'] / r[1]
            S_f_perpendicular_norm_z = params['S_f_perpendicular'] / r[2]
            
            # Shear strength parameters
            S_f_shear_xy_norm = params['S_f_shear'] / r[3]
            S_f_shear_yz_norm = params['S_f_shear'] / r[4]
            S_f_shear_xz_norm = params['S_f_shear'] / r[5]
            
            transformed[case_id] = {
                'S_f_parallel_t_norm': torch.tensor(float(S_f_parallel_t_norm), dtype=torch.float32, device=self.device),
                'S_f_parallel_c_norm': torch.tensor(float(S_f_parallel_c_norm), dtype=torch.float32, device=self.device),
                'S_f_perpendicular_norm_y': torch.tensor(float(S_f_perpendicular_norm_y), dtype=torch.float32, device=self.device),
                'S_f_perpendicular_norm_z': torch.tensor(float(S_f_perpendicular_norm_z), dtype=torch.float32, device=self.device),
                'S_f_shear_xy_norm': torch.tensor(float(S_f_shear_xy_norm), dtype=torch.float32, device=self.device),
                'S_f_shear_yz_norm': torch.tensor(float(S_f_shear_yz_norm), dtype=torch.float32, device=self.device),
                'S_f_shear_xz_norm': torch.tensor(float(S_f_shear_xz_norm), dtype=torch.float32, device=self.device),
                
                # Keep dimensionless parameters
                'eta_parallel': params['eta_parallel'],
                'eta_perpendicular': params['eta_perpendicular']
            }
            
            if case_id == 1:
                print(f"  Case {case_id} Cuntze transformation example:")
                print(f"    Fiber tensile strength: {params['S_f_parallel_t']:.0f} -> {S_f_parallel_t_norm:.4f}")
                print(f"    Fiber compressive strength: {params['S_f_parallel_c']:.0f} -> {S_f_parallel_c_norm:.4f}")
        
        print("✅ Cuntze parameter transformation completed")
        return transformed

    def _compute_cuntze_loss_normalized(self, stress_normalized, case_ids_batch):
        """Compute Cuntze criterion physics loss in normalized space"""
        batch_size = stress_normalized.shape[0]
        total_loss = torch.tensor(0.0, device=self.device)
        
        for i in range(batch_size):
            case_id = int(case_ids_batch[i].item())
            if case_id not in self.normalized_params:
                case_id = 1  # Use default case
            
            params = self.normalized_params[case_id]
            s = stress_normalized[i]  # [σx_norm, σy_norm, σz_norm, τxy_norm, τyz_norm, τxz_norm]
            
            # Cuntze criterion: fiber failure and matrix failure
            
            # 1. Fiber tensile failure (σx > 0)
            fiber_tension_criterion = torch.max(
                torch.tensor(0.0, device=self.device),
                s[0] / params['S_f_parallel_t_norm'] - 1.0
            )
            
            # 2. Fiber compression failure (σx < 0)
            fiber_compression_criterion = torch.max(
                torch.tensor(0.0, device=self.device),
                -s[0] / params['S_f_parallel_c_norm'] - 1.0
            )
            
            # 3. Matrix failure (combination of transverse and shear stresses)
            # Simplified matrix failure criterion
            matrix_criterion_y = torch.max(
                torch.tensor(0.0, device=self.device),
                torch.abs(s[1]) / params['S_f_perpendicular_norm_y'] - 1.0
            )
            
            matrix_criterion_z = torch.max(
                torch.tensor(0.0, device=self.device),
                torch.abs(s[2]) / params['S_f_perpendicular_norm_z'] - 1.0
            )
            
            # 4. Shear failure
            shear_criterion_xy = torch.max(
                torch.tensor(0.0, device=self.device),
                torch.abs(s[3]) / params['S_f_shear_xy_norm'] - 1.0
            )
            
            shear_criterion_yz = torch.max(
                torch.tensor(0.0, device=self.device),
                torch.abs(s[4]) / params['S_f_shear_yz_norm'] - 1.0
            )
            
            shear_criterion_xz = torch.max(
                torch.tensor(0.0, device=self.device),
                torch.abs(s[5]) / params['S_f_shear_xz_norm'] - 1.0
            )
            
            # Combine all failure modes
            total_criterion = (
                fiber_tension_criterion**2 + 
                fiber_compression_criterion**2 + 
                matrix_criterion_y**2 + 
                matrix_criterion_z**2 + 
                shear_criterion_xy**2 + 
                shear_criterion_yz**2 + 
                shear_criterion_xz**2
            )
            
            total_loss += total_criterion
        
        return total_loss / batch_size

    def _compute_cuntze_loss_original(self, stress_original, case_ids_batch):
        """Compute Cuntze criterion physics loss in original space"""
        batch_size = stress_original.shape[0]
        total_loss = torch.tensor(0.0, device=self.device)
        
        for i in range(batch_size):
            case_id = int(case_ids_batch[i].item())
            if case_id not in self.original_params:
                case_id = 1
            
            params = self.original_params[case_id]
            s = stress_original[i]
            
            # Cuntze criterion based on original strength values
            fiber_tension = torch.max(
                torch.tensor(0.0, device=self.device),
                s[0] / params['S_f_parallel_t'] - 1.0
            )
            
            fiber_compression = torch.max(
                torch.tensor(0.0, device=self.device),
                -s[0] / params['S_f_parallel_c'] - 1.0
            )
            
            matrix_failure = torch.max(
                torch.tensor(0.0, device=self.device),
                torch.sqrt(s[1]**2 + s[2]**2) / params['S_f_perpendicular'] - 1.0
            )
            
            shear_failure = torch.max(
                torch.tensor(0.0, device=self.device),
                torch.sqrt(s[3]**2 + s[4]**2 + s[5]**2) / params['S_f_shear'] - 1.0
            )
            
            total_criterion = (
                fiber_tension**2 + 
                fiber_compression**2 + 
                matrix_failure**2 + 
                shear_failure**2
            )
            
            total_loss += total_criterion
        
        return total_loss / batch_size

    def _convert_to_normalized_space(self, stress_original):
        """Convert stress from original scale to normalized space"""
        if self.use_normalized_params:
            return (stress_original - self.stress_min.unsqueeze(0)) / self.stress_range.unsqueeze(0)
        else:
            return stress_original

    def compute_cuntze_loss(self, stress_pred, case_ids_batch):
        """Compute Cuntze criterion physics loss"""
        if self.use_normalized_params:
            stress_for_physics = self._convert_to_normalized_space(stress_pred)
            loss_cuntze = self._compute_cuntze_loss_normalized(stress_for_physics, case_ids_batch)
        else:
            loss_cuntze = self._compute_cuntze_loss_original(stress_pred, case_ids_batch)
        
        return loss_cuntze

    def compute_all_cuntze_loss(self, stress_pred, stress_true, case_ids_batch, lambda_weight, W_cuntze):
        """Compute complete Cuntze loss"""
        
        # 1. Data loss
        loss_data = F.mse_loss(stress_pred, stress_true)
        
        # 2. Cuntze physics loss
        if lambda_weight > 0 and W_cuntze > 0:
            loss_cuntze = self.compute_cuntze_loss(stress_pred, case_ids_batch)
        else:
            loss_cuntze = torch.tensor(0.0, device=self.device)
        
        # 3. Total loss
        total_loss = loss_data + lambda_weight * W_cuntze * loss_cuntze
        
        return total_loss, loss_data, loss_cuntze

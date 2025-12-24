# physics_loss.py
# Responsibility: Encapsulate and calculate physics loss based on LaRC criterion.

import torch
import torch.nn.functional as F
import numpy as np

class LaRCPhysicsLoss:
    """Physics loss calculator based on LaRC criterion - supports decoupled prediction mode"""
    
    def __init__(self, larc_coeffs, device, processor=None):
        self.device = device
        self.processor = processor
        
        # Keep original LaRC coefficients (only for validation and visualization)
        self.original_coeffs = {}
        for case_id, coeffs in larc_coeffs.items():
            self.original_coeffs[case_id] = {
                k: torch.tensor(v, dtype=torch.float32, device=device) 
                for k, v in coeffs.items()
            }
        
        # Transformation coefficients in normalized space
        self.normalized_coeffs = {}
        self.use_normalized_coeffs = False
        
        # Check if in decoupled prediction mode and create dummy stress normalization parameters
        if processor and processor.is_fitted:
            if hasattr(processor, 'use_L_normalization') and processor.use_L_normalization:
                # Decoupled prediction mode: create normalization parameters from original stress data
                if hasattr(processor, 'original_stress_tensor'):
                    print("🔧 Decoupled prediction mode: creating normalization parameters from original stress data...")
                    self._create_stress_normalization_from_original_data(processor)
                else:
                    print("⚠️ Decoupled prediction mode: cannot find original stress data, using original coefficients")
            elif hasattr(processor, 'use_stress_normalization') and processor.use_stress_normalization:
                # Traditional mode: use existing stress normalization parameters
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
                    print(f"⚠️ Stress component {i} range too small, set to default value")
            
            # Create normalization parameters
            self.stress_min = torch.tensor(stress_min, dtype=torch.float32, device=self.device)
            self.stress_range = torch.tensor(stress_range, dtype=torch.float32, device=self.device)
            
            # Transform Tsai-Wu coefficients to normalized space
            self.normalized_coeffs = self._transform_coefficients_to_normalized_space(self.original_coeffs)
            self.use_normalized_coeffs = True
            
            print("✅ Decoupled prediction mode: stress normalization parameters created successfully")
            print(f"📊 Stress range: min={stress_min}, max={stress_max}")
            
        except Exception as e:
            print(f"❌ Failed to create stress normalization parameters: {e}")
            self.use_normalized_coeffs = False

    def _load_existing_stress_normalization(self, processor):
        """Load existing stress normalization parameters in traditional mode"""
        try:
            if hasattr(processor, 'stress_scaler') and hasattr(processor.stress_scaler, 'data_min_'):
                self.stress_min = torch.tensor(processor.stress_scaler.data_min_, dtype=torch.float32, device=self.device)
                self.stress_range = torch.tensor(processor.stress_scaler.data_range_, dtype=torch.float32, device=self.device)
                self.normalized_coeffs = self._transform_coefficients_to_normalized_space(self.original_coeffs)
                self.use_normalized_coeffs = True
                print("✅ Traditional mode: existing stress normalization parameters loaded")
            else:
                print("⚠️ Traditional mode: valid stress normalization parameters not found")
                self.use_normalized_coeffs = False
        except Exception as e:
            print(f"❌ Failed to load stress normalization parameters: {e}")
            self.use_normalized_coeffs = False

    def _transform_coefficients_to_normalized_space(self, original_coeffs):
        """Transform LaRC material parameters to fully normalized space"""
        transformed = {}
        
        print("🔧 Starting transformation of LaRC material parameters to fully normalized space...")
        
        for case_id, coeffs in original_coeffs.items():
            # Normalization parameters
            r = self.stress_range  # [range_x, range_y, range_z, range_xy, range_yz, range_xz]
            m = self.stress_min    # [min_x, min_y, min_z, min_xy, min_yz, min_xz]
            
            # For LaRC criterion, we need to transform all strength parameters to normalized stress space
            # This ensures all parameters are consistent when performing LaRC criterion judgment in normalized space
            
            # Fiber strength (mainly x direction) - normalized to [0,1] space
            X_T_norm = coeffs['X_T'] / r[0] if r[0] > 1e-8 else 1.0
            X_C_norm = coeffs['X_C'] / r[0] if r[0] > 1e-8 else 1.0
            
            # Matrix strength (mainly y, z directions) - normalized to [0,1] space
            Y_T_norm = coeffs['Y_T'] / r[1] if r[1] > 1e-8 else 1.0
            Y_C_norm = coeffs['Y_C'] / r[1] if r[1] > 1e-8 else 1.0
            
            # Shear strength - normalized to [0,1] space
            S_L_norm = coeffs['S_L'] / r[3] if r[3] > 1e-8 else 1.0  # In-plane shear τxy
            S_T_norm = coeffs['S_T'] / r[4] if r[4] > 1e-8 else 1.0  # Out-of-plane shear τyz
            S_12_norm = coeffs['S_12'] / r[3] if r[3] > 1e-8 else 1.0
            
            # Shear modulus - special handling as its units in LaRC formula
            # G_12 is used for fiber micro-buckling, its unit should be consistent with stress
            G_12_norm = coeffs['G_12'] / r[0] if r[0] > 1e-8 else 1.0
            
            # Friction coefficients - dimensionless, but need to adjust their effect based on normalized stress
            # Friction coefficient multiplies normal stress, so stress normalization effect needs consideration
            eta_T_adjusted = coeffs['eta_T']  # Keep original value as it multiplies normalized stress
            eta_L_adjusted = coeffs['eta_L']  # Keep original value as it multiplies normalized stress
            
            # To ensure numerical stability, set minimum threshold
            min_threshold = 1e-6
            
            transformed[case_id] = {
                # Fiber strength parameters (normalized)
                'X_T_norm': torch.tensor(max(X_T_norm, min_threshold), dtype=torch.float32, device=self.device),
                'X_C_norm': torch.tensor(max(X_C_norm, min_threshold), dtype=torch.float32, device=self.device),
                
                # Matrix strength parameters (normalized)
                'Y_T_norm': torch.tensor(max(Y_T_norm, min_threshold), dtype=torch.float32, device=self.device),
                'Y_C_norm': torch.tensor(max(Y_C_norm, min_threshold), dtype=torch.float32, device=self.device),
                
                # Shear strength parameters (normalized)
                'S_L_norm': torch.tensor(max(S_L_norm, min_threshold), dtype=torch.float32, device=self.device),
                'S_T_norm': torch.tensor(max(S_T_norm, min_threshold), dtype=torch.float32, device=self.device),
                'S_12_norm': torch.tensor(max(S_12_norm, min_threshold), dtype=torch.float32, device=self.device),
                
                # Shear modulus (normalized)
                'G_12_norm': torch.tensor(max(G_12_norm, min_threshold), dtype=torch.float32, device=self.device),
                
                # Friction coefficients (adjusted)
                'eta_T': torch.tensor(eta_T_adjusted, dtype=torch.float32, device=self.device),
                'eta_L': torch.tensor(eta_L_adjusted, dtype=torch.float32, device=self.device),
                
                # Original values for comparison
                'X_T_orig': torch.tensor(coeffs['X_T'], dtype=torch.float32, device=self.device),
                'Y_T_orig': torch.tensor(coeffs['Y_T'], dtype=torch.float32, device=self.device),
                'S_L_orig': torch.tensor(coeffs['S_L'], dtype=torch.float32, device=self.device),
                
                # Normalization factors for debugging
                'norm_factor_x': torch.tensor(r[0], dtype=torch.float32, device=self.device),
                'norm_factor_y': torch.tensor(r[1], dtype=torch.float32, device=self.device),
                'norm_factor_shear': torch.tensor(r[3], dtype=torch.float32, device=self.device),
            }
            
            if case_id == 1:
                print(f"  Case {case_id} LaRC parameters fully normalized transformation:")
                print(f"    X_T: {coeffs['X_T']:.0f} MPa -> {X_T_norm:.6f} (normalized)")
                print(f"    Y_T: {coeffs['Y_T']:.0f} MPa -> {Y_T_norm:.6f} (normalized)")
                print(f"    S_L: {coeffs['S_L']:.0f} MPa -> {S_L_norm:.6f} (normalized)")
                print(f"    G_12: {coeffs['G_12']:.0f} MPa -> {G_12_norm:.6f} (normalized)")
                print(f"    Stress range: σx={r[0]:.2f}, σy={r[1]:.2f}, τxy={r[3]:.2f}")
                print(f"    Friction coefficients: ηT={eta_T_adjusted:.3f}, ηL={eta_L_adjusted:.3f} (kept)")
        
        print("✅ LaRC material parameters fully normalized transformation completed")
        return transformed

    def _compute_larc_loss_normalized(self, stress_normalized, case_ids_batch):
        """Directly compute LaRC criterion physics loss in fully normalized space - optimized magnitude version"""
        batch_size = stress_normalized.shape[0]
        total_loss = torch.tensor(0.0, device=self.device)
        valid_samples = 0
        
        for i in range(batch_size):
            case_id = int(case_ids_batch[i].item())
            if case_id not in self.normalized_coeffs:
                case_id = 1  # Use default case
            
            coeffs = self.normalized_coeffs[case_id]
            s = stress_normalized[i]  # [σx_norm, σy_norm, σz_norm, τxy_norm, τyz_norm, τxz_norm]
            
            # Input check: ensure stress is in reasonable range
            if torch.any(torch.isnan(s)) or torch.any(torch.isinf(s)):
                continue
            
            # Compute LaRC criterion failure indices
            try:
                larc_violations = self._compute_larc_failure_indices(s, coeffs)
                
                if not larc_violations:  # If no valid failure indices
                    continue
                
                # Compute physics loss using moderate loss function
                sample_violation = 0.0
                violation_count = 0
                
                for failure_index in larc_violations:
                    if torch.isnan(failure_index) or torch.isinf(failure_index):
                        continue
                    
                    # Only compute loss when failure index > 1
                    if failure_index > 1.0:
                        # Use square root loss function with moderate growth
                        excess = failure_index - 1.0
                        
                        # Improved loss function: neither too aggressive nor too mild
                        if excess <= 1.0:
                            violation = excess ** 1.5  # Growth slightly slower than quadratic
                        else:
                            violation = 1.0 + 0.5 * excess  # Linear growth
                        
                        sample_violation += violation
                        violation_count += 1
                
                if violation_count > 0:
                    # Increase base magnitude of physics loss to participate in training
                    sample_violation = sample_violation * 0.5  # Moderate scaling factor
                
                # Set reasonable upper limit but not too low
                sample_violation = torch.clamp(sample_violation, 0.0, 5.0)
                total_loss += sample_violation
                valid_samples += 1
                
            except Exception as e:
                continue
        
        # If no valid samples, return zero loss
        if valid_samples == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Average loss
        avg_loss = total_loss / valid_samples
        
        # Ensure physics loss has base magnitude to avoid being too small
        if avg_loss.item() > 0:
            avg_loss = torch.max(avg_loss, torch.tensor(0.001, device=self.device))
        
        # Final numerical check
        if torch.isnan(avg_loss) or torch.isinf(avg_loss):
            return torch.tensor(0.0, device=self.device)
        
        return avg_loss

    def _compute_larc_failure_indices(self, stress, coeffs):
        """Compute failure indices for various failure modes according to LaRC criterion - in fully normalized space"""
        # Stress components (already normalized to [0,1] space)
        sigma_x, sigma_y, sigma_z = stress[0], stress[1], stress[2]
        tau_xy, tau_yz, tau_xz = stress[3], stress[4], stress[5]
        
        failure_indices = []
        
        # 1. Fiber tensile failure (σx ≥ 0) - in normalized space
        if sigma_x >= 0:
            # Failure criterion in normalized space: σx_norm / X_T_norm ≤ 1
            I_F_plus = sigma_x / coeffs['X_T_norm']
            failure_indices.append(I_F_plus)
            
            # Debug info
            if torch.isnan(I_F_plus) or torch.isinf(I_F_plus):
                print(f"⚠️ Fiber tensile failure index anomaly: σx={sigma_x:.6f}, X_T_norm={coeffs['X_T_norm']:.6f}")
        
        # 2. Fiber compressive failure (σx < 0) - fiber micro-buckling in normalized space
        if sigma_x < 0:
            # LaRC04 fiber compression failure formula (normalized version)
            X_C_norm = coeffs['X_C_norm']
            G_12_norm = coeffs['G_12_norm']
            
            # Avoid division by zero and numerical instability
            if G_12_norm > 1e-6 and X_C_norm > 1e-6:
                # Micro-buckling failure formula in normalized space
                ratio = X_C_norm / (2 * G_12_norm)
                if ratio < 100:  # Avoid numerical overflow
                    denominator = X_C_norm - (X_C_norm**2) / (4 * G_12_norm) * (1 / (1 + ratio))
                    if abs(denominator) > 1e-6:
                        I_F_minus = (-sigma_x) / denominator
                        failure_indices.append(I_F_minus)
                        
                        # Debug info
                        if torch.isnan(I_F_minus) or torch.isinf(I_F_minus):
                            print(f"⚠️ Fiber compression failure index anomaly: σx={sigma_x:.6f}, denominator={denominator:.6f}")
        
        # 3. Matrix failure mode - in normalized space considering most dangerous fracture plane
        try:
            matrix_failure_index = self._compute_matrix_failure_index(stress, coeffs)
            if matrix_failure_index is not None and not torch.isnan(matrix_failure_index) and not torch.isinf(matrix_failure_index):
                failure_indices.append(matrix_failure_index)
            elif matrix_failure_index is not None:
                print(f"⚠️ Matrix failure index anomaly: {matrix_failure_index}")
        except Exception as e:
            print(f"⚠️ Matrix failure calculation error: {e}")
        
        return failure_indices

    def _compute_matrix_failure_index(self, stress, coeffs):
        """Compute matrix failure index - considering most dangerous fracture plane"""
        sigma_x, sigma_y, sigma_z = stress[0], stress[1], stress[2]
        tau_xy, tau_yz, tau_xz = stress[3], stress[4], stress[5]
        
        # Compute most dangerous fracture angle according to LaRC criterion
        theta_0 = self._compute_critical_fracture_angle(stress, coeffs)
        
        # Compute stresses on fracture plane
        sigma_n, tau_T, tau_L = self._compute_fracture_plane_stresses(stress, theta_0)
        
        # Select failure criterion based on normal stress sign
        if sigma_n >= 0:
            # Matrix tensile and shear failure
            I_M_plus = self._compute_matrix_tensile_failure(sigma_n, tau_T, tau_L, coeffs)
            return I_M_plus
        else:
            # Matrix compressive failure
            I_M_minus = self._compute_matrix_compressive_failure(sigma_n, tau_T, tau_L, coeffs)
            return I_M_minus

    def _compute_critical_fracture_angle(self, stress, coeffs):
        """Compute most dangerous fracture angle θ_0 - in normalized space"""
        sigma_x, sigma_y, sigma_z = stress[0], stress[1], stress[2]
        tau_xy, tau_yz, tau_xz = stress[3], stress[4], stress[5]
        
        # Compute fracture angle according to LaRC criterion simplified formula (normalized space version)
        S_L_norm = coeffs['S_L_norm']
        S_T_norm = coeffs['S_T_norm']
        Y_T_norm = coeffs['Y_T_norm']
        
        # Numerical stability check
        min_threshold = 1e-6
        if S_L_norm < min_threshold or S_T_norm < min_threshold or Y_T_norm < min_threshold:
            return torch.tensor(0.0, device=self.device)  # Default angle
        
        # Compute I_x and I_z (in normalized space)
        I_x = (tau_xy / S_L_norm) ** 2
        I_z = (sigma_y / Y_T_norm) ** 2 + (tau_yz / S_T_norm) ** 2
        
        # Avoid zero denominator
        denominator = I_x + I_z
        if denominator < min_threshold:
            return torch.tensor(0.0, device=self.device)
        
        # Safe square root computation
        cos_theta_0_sq = I_x / denominator
        sin_theta_0_sq = I_z / denominator
        
        # Ensure values in valid range
        cos_theta_0_sq = torch.clamp(cos_theta_0_sq, 0.0, 1.0)
        sin_theta_0_sq = torch.clamp(sin_theta_0_sq, 0.0, 1.0)
        
        cos_theta_0 = torch.sqrt(cos_theta_0_sq)
        sin_theta_0 = torch.sqrt(sin_theta_0_sq) * torch.sign(tau_yz)
        
        theta_0 = torch.atan2(sin_theta_0, cos_theta_0)
        
        # Numerical check
        if torch.isnan(theta_0) or torch.isinf(theta_0):
            print(f"⚠️ Fracture angle calculation anomaly: I_x={I_x:.6f}, I_z={I_z:.6f}")
            return torch.tensor(0.0, device=self.device)
        
        return theta_0

    def _compute_fracture_plane_stresses(self, stress, theta):
        """Compute stress components on fracture plane"""
        sigma_x, sigma_y, sigma_z = stress[0], stress[1], stress[2]
        tau_xy, tau_yz, tau_xz = stress[3], stress[4], stress[5]
        
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        cos_2theta = torch.cos(2 * theta)
        sin_2theta = torch.sin(2 * theta)
        
        # Normal stress on fracture plane
        sigma_n = (sigma_y + sigma_z) / 2 + (sigma_y - sigma_z) / 2 * cos_2theta + tau_yz * sin_2theta
        
        # Transverse shear stress on fracture plane
        tau_T = -(sigma_y - sigma_z) / 2 * sin_2theta + tau_yz * cos_2theta
        
        # Longitudinal shear stress on fracture plane
        tau_L = tau_xy * cos_theta - tau_xz * sin_theta
        
        return sigma_n, tau_T, tau_L

    def _compute_matrix_tensile_failure(self, sigma_n, tau_T, tau_L, coeffs):
        """Compute matrix tensile failure index - in normalized space with improved numerical stability"""
        S_T_norm = coeffs['S_T_norm']
        S_L_norm = coeffs['S_L_norm']
        eta_T = coeffs['eta_T']
        eta_L = coeffs['eta_L']
        
        # Improved numerical stability check
        min_threshold = 1e-8  # Lower threshold
        
        # Compute corrected strength (considering friction effect)
        # In normalized space, friction terms need careful handling
        denom_T = S_T_norm - eta_T * sigma_n
        denom_L = S_L_norm - eta_L * sigma_n
        
        # More reasonable numerical checks
        if S_T_norm < min_threshold or S_L_norm < min_threshold:
            return torch.tensor(0.0, device=self.device)  # Invalid material parameters
        
        # Check if friction term causes denominator to be too small
        if abs(denom_T) < min_threshold:
            # When friction term approaches strength, use approximation
            denom_T = torch.sign(denom_T) * min_threshold if denom_T != 0 else min_threshold
        
        if abs(denom_L) < min_threshold:
            denom_L = torch.sign(denom_L) * min_threshold if denom_L != 0 else min_threshold
        
        # Compute failure indices but limit to reasonable range
        term_T = (tau_T / denom_T) ** 2
        term_L = (tau_L / denom_L) ** 2
        
        # Limit single items to avoid numerical explosion
        term_T = torch.clamp(term_T, 0.0, 25.0)  # Limit single item max failure index to 5²
        term_L = torch.clamp(term_L, 0.0, 25.0)
        
        I_M_plus = term_T + term_L
        
        # Final check
        if torch.isnan(I_M_plus) or torch.isinf(I_M_plus):
            return torch.tensor(0.0, device=self.device)
        
        return torch.clamp(I_M_plus, 0.0, 50.0)  # Limit total failure index

    def _compute_matrix_compressive_failure(self, sigma_n, tau_T, tau_L, coeffs):
        """Compute matrix compressive failure index - in normalized space with improved numerical stability"""
        S_T_norm = coeffs['S_T_norm']
        S_L_norm = coeffs['S_L_norm']
        Y_C_norm = coeffs['Y_C_norm']
        eta_T = coeffs['eta_T']
        eta_L = coeffs['eta_L']
        
        min_threshold = 1e-8
        
        # Material parameter validity check
        if S_T_norm < min_threshold or S_L_norm < min_threshold or Y_C_norm < min_threshold:
            return torch.tensor(0.0, device=self.device)
        
        # Compute corrected strength
        denom_T = S_T_norm - eta_T * sigma_n
        denom_L = S_L_norm - eta_L * sigma_n
        
        # Improved denominator check
        if abs(denom_T) < min_threshold:
            denom_T = torch.sign(denom_T) * min_threshold if denom_T != 0 else min_threshold
        
        if abs(denom_L) < min_threshold:
            denom_L = torch.sign(denom_L) * min_threshold if denom_L != 0 else min_threshold
        
        # Compute by components with range limiting
        term_T = torch.clamp((tau_T / denom_T) ** 2, 0.0, 25.0)
        term_L = torch.clamp((tau_L / denom_L) ** 2, 0.0, 25.0)
        term_N = torch.clamp((sigma_n / Y_C_norm) ** 2, 0.0, 25.0)
        
        I_M_minus = term_T + term_L + term_N
        
        # Final check
        if torch.isnan(I_M_minus) or torch.isinf(I_M_minus):
            return torch.tensor(0.0, device=self.device)
        
        return torch.clamp(I_M_minus, 0.0, 50.0)

    def _compute_larc_loss_original(self, stress_original, case_ids_batch):
        """Compute LaRC criterion physics loss in original space (for comparison only)"""
        batch_size = stress_original.shape[0]
        total_loss = torch.tensor(0.0, device=self.device)
        
        for i in range(batch_size):
            case_id = int(case_ids_batch[i].item())
            if case_id not in self.original_coeffs:
                case_id = 1
            
            coeffs = self.original_coeffs[case_id]
            s = stress_original[i]
            
            # Compute LaRC criterion failure indices
            larc_violations = self._compute_larc_failure_indices_original(s, coeffs)
            
            # Failure indices should be ≤ 1, loss produced when violated
            total_violation = 0.0
            for failure_index in larc_violations:
                if failure_index > 1.0:
                    total_violation += (failure_index - 1.0) ** 2
            
            total_loss += total_violation
        
        return total_loss / batch_size

    def _compute_larc_failure_indices_original(self, stress, coeffs):
        """Compute failure indices for various failure modes according to LaRC criterion in original space"""
        # Stress components
        sigma_x, sigma_y, sigma_z = stress[0], stress[1], stress[2]
        tau_xy, tau_yz, tau_xz = stress[3], stress[4], stress[5]
        
        failure_indices = []
        
        # 1. Fiber tensile failure (σx ≥ 0)
        if sigma_x >= 0:
            I_F_plus = sigma_x / coeffs['X_T']
            failure_indices.append(I_F_plus)
        
        # 2. Fiber compressive failure (σx < 0) - fiber micro-buckling
        if sigma_x < 0:
            X_C = coeffs['X_C']
            G_12 = coeffs['G_12']
            
            if G_12 > 1e-8:
                denominator = X_C - (X_C**2) / (4 * G_12) * (1 / (1 + X_C / (2 * G_12)))
                if abs(denominator) > 1e-8:
                    I_F_minus = (-sigma_x) / denominator
                    failure_indices.append(I_F_minus)
        
        # 3. Matrix failure mode - use original parameters
        matrix_failure_index = self._compute_matrix_failure_index_original(stress, coeffs)
        if matrix_failure_index is not None:
            failure_indices.append(matrix_failure_index)
        
        return failure_indices

    def _compute_matrix_failure_index_original(self, stress, coeffs):
        """Compute matrix failure index in original space"""
        sigma_x, sigma_y, sigma_z = stress[0], stress[1], stress[2]
        tau_xy, tau_yz, tau_xz = stress[3], stress[4], stress[5]
        
        # Compute most dangerous fracture angle
        theta_0 = self._compute_critical_fracture_angle_original(stress, coeffs)
        
        # Compute stresses on fracture plane
        sigma_n, tau_T, tau_L = self._compute_fracture_plane_stresses_original(stress, theta_0)
        
        # Select failure criterion based on normal stress sign
        if sigma_n >= 0:
            return self._compute_matrix_tensile_failure_original(sigma_n, tau_T, tau_L, coeffs)
        else:
            return self._compute_matrix_compressive_failure_original(sigma_n, tau_T, tau_L, coeffs)

    def _compute_critical_fracture_angle_original(self, stress, coeffs):
        """Compute most dangerous fracture angle in original space"""
        sigma_x, sigma_y, sigma_z = stress[0], stress[1], stress[2]
        tau_xy, tau_yz, tau_xz = stress[3], stress[4], stress[5]
        
        S_L = coeffs['S_L']
        S_T = coeffs['S_T']
        Y_T = coeffs['Y_T']
        
        if abs(S_L) < 1e-8 or abs(S_T) < 1e-8 or abs(Y_T) < 1e-8:
            return 0.0
        
        I_x = (tau_xy / S_L) ** 2
        I_z = (sigma_y / Y_T) ** 2 + (tau_yz / S_T) ** 2
        
        denominator = I_x + I_z
        if denominator < 1e-8:
            return 0.0
        
        cos_theta_0 = torch.sqrt(I_x / denominator)
        sin_theta_0 = torch.sqrt(I_z / denominator) * torch.sign(tau_yz)
        
        return torch.atan2(sin_theta_0, cos_theta_0)

    def _compute_fracture_plane_stresses_original(self, stress, theta):
        """Compute stress components on fracture plane in original space"""
        sigma_x, sigma_y, sigma_z = stress[0], stress[1], stress[2]
        tau_xy, tau_yz, tau_xz = stress[3], stress[4], stress[5]
        
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        cos_2theta = torch.cos(2 * theta)
        sin_2theta = torch.sin(2 * theta)
        
        sigma_n = (sigma_y + sigma_z) / 2 + (sigma_y - sigma_z) / 2 * cos_2theta + tau_yz * sin_2theta
        tau_T = -(sigma_y - sigma_z) / 2 * sin_2theta + tau_yz * cos_2theta
        tau_L = tau_xy * cos_theta - tau_xz * sin_theta
        
        return sigma_n, tau_T, tau_L

    def _compute_matrix_tensile_failure_original(self, sigma_n, tau_T, tau_L, coeffs):
        """Compute matrix tensile failure index in original space"""
        S_T = coeffs['S_T']
        S_L = coeffs['S_L']
        eta_T = coeffs['eta_T']
        eta_L = coeffs['eta_L']
        
        denom_T = S_T - eta_T * sigma_n
        denom_L = S_L - eta_L * sigma_n
        
        if abs(denom_T) < 1e-8 or abs(denom_L) < 1e-8:
            return torch.tensor(0.0, device=self.device)
        
        return (tau_T / denom_T) ** 2 + (tau_L / denom_L) ** 2

    def _compute_matrix_compressive_failure_original(self, sigma_n, tau_T, tau_L, coeffs):
        """Compute matrix compressive failure index in original space"""
        S_T = coeffs['S_T']
        S_L = coeffs['S_L']
        Y_C = coeffs['Y_C']
        eta_T = coeffs['eta_T']
        eta_L = coeffs['eta_L']
        
        denom_T = S_T - eta_T * sigma_n
        denom_L = S_L - eta_L * sigma_n
        
        if abs(denom_T) < 1e-8 or abs(denom_L) < 1e-8 or abs(Y_C) < 1e-8:
            return torch.tensor(0.0, device=self.device)
        
        return (tau_T / denom_T) ** 2 + (tau_L / denom_L) ** 2 + (sigma_n / Y_C) ** 2

    def _compute_L_monitor_normalized(self, stress_pred_normalized, stress_true_normalized):
        """Compute failure length monitoring loss in normalized space"""
        L_pred = torch.sqrt(torch.sum(stress_pred_normalized**2, dim=1))
        L_true = torch.sqrt(torch.sum(stress_true_normalized**2, dim=1))
        return F.mse_loss(L_pred, L_true)

    def compute_all_normalized_loss(self, stress_pred_normalized, stress_true_normalized, 
                                   case_ids_batch, lambda_weight, W_physics):
        """Compute all losses in normalized space - supports decoupled prediction mode"""
        
        # 1. Data loss - for decoupled prediction, compute loss of reconstructed stress tensor
        loss_data_stress = F.mse_loss(stress_pred_normalized, stress_true_normalized)
        
        # 2. Physics loss - use reconstructed stress tensor
        if lambda_weight > 0 and W_physics > 0:
            if self.use_normalized_coeffs:
                # Need to transform stress tensor from original scale to normalized scale
                stress_for_physics = self._convert_to_normalized_space(stress_pred_normalized)
                loss_physics = self._compute_larc_loss_normalized(stress_for_physics, case_ids_batch)
            else:
                # If no normalized coefficients, compute in original space
                loss_physics = self._compute_larc_loss_original(stress_pred_normalized, case_ids_batch)
        else:
            loss_physics = torch.tensor(0.0, device=self.device)
        
        # 3. Total loss
        total_loss = loss_data_stress + lambda_weight * W_physics * loss_physics
        
        # 4. Monitoring metrics
        loss_L_monitor = self._compute_L_monitor_normalized(stress_pred_normalized, stress_true_normalized)
        
        return total_loss, loss_data_stress, loss_physics, loss_L_monitor

    def _convert_to_normalized_space(self, stress_original):
        """Transform stress from original scale to normalized space"""
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
                loss_physics = self._compute_larc_loss_normalized(stress_pred_normalized, case_ids_batch)
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

    # Maintain backward compatible interface
    def compute_fixed_weight_loss(self, stress_pred_scaled, stress_true_scaled, 
                                 case_ids_batch, lambda_weight, W_physics):
        """Backward compatible interface - redirect to normalized loss computation"""
        return self.compute_all_normalized_loss(stress_pred_scaled, stress_true_scaled, 
                                               case_ids_batch, lambda_weight, W_physics)

    def compute_stable_adaptive_loss(self, stress_pred_scaled, stress_true_scaled, 
                                   case_ids_batch, lambda_weight, alpha=0.1):
        """Backward compatible interface - redirect to normalized adaptive loss computation"""
        return self.compute_adaptive_normalized_loss(stress_pred_scaled, stress_true_scaled, 
                                                    case_ids_batch, lambda_weight, alpha)
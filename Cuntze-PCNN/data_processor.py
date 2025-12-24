# data_processor.py
# Responsibility: Handle all data loading, preprocessing, normalization, and dataset splitting.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DualConstraintDataProcessor:
    """Dual-constraint PINN data processor"""
    
    def __init__(self):
        """
        Initialize data processor
        
        Args:
            use_normalization: Whether to enable data normalization
        """
        self.feature_scaler = MinMaxScaler()  # Normalize 20-dimensional enhanced features
        self.L_scaler = MinMaxScaler()        # Normalize 1-dimensional failure length L
        self.is_fitted = False
        self.feature_names = None
        self.original_data = None
        self.use_L_normalization = True      # Enable L value standardization
        
    def load_data(self, data_file='datasetnew.csv'):
        """Load data"""
        print(f"🔧 Dual-constraint mode: Loading data from {data_file}...")
        df = pd.read_csv(data_file)
        self.original_data = df.copy()
        
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        required_stress_columns = ['sx', 'sy', 'sz', 'txy', 'tyz', 'txz']
        missing_columns = [col for col in required_stress_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required stress columns: {missing_columns}")
        
        return df
    
    def prepare_decoupled_features_and_labels(self, df):
        """Prepare features and labels for dual-constraint learning"""
        print("🚀 Dual-constraint mode: Preparing 20-dimensional enhanced features and 1-dimensional failure length label...")
        
        stress_columns = ['sx', 'sy', 'sz', 'txy', 'tyz', 'txz']
        material_columns = [col for col in df.columns if col not in stress_columns + ['case_id']]
        
        # Take first 14 material properties as base features
        if len(material_columns) < 14:
            raise ValueError(f"Requires at least 14 material properties, got {len(material_columns)}")
        
        base_features = df[material_columns[:14]]  # 14-dimensional base material features
        stress_tensor = df[stress_columns]         # 6-dimensional stress tensor
        
        print(f"✅ Base material features (14-dim): {base_features.columns.tolist()}")
        
        # Data quality check
        print(f"\n📊 Original data quality check:")
        print(f"  Base features shape: {base_features.shape}")
        print(f"  Stress tensor shape: {stress_tensor.shape}")
        print(f"  Base features missing values: {base_features.isnull().sum().sum()}")
        print(f"  Stress tensor missing values: {stress_tensor.isnull().sum().sum()}")
        
        # Filter zero stress states
        stress_magnitude = np.sqrt(np.sum(stress_tensor**2, axis=1))
        valid_mask = stress_magnitude > 1e-6
        
        print(f"\n🔍 Data filtering analysis:")
        print(f"  Stress magnitude range: [{stress_magnitude.min():.2e}, {stress_magnitude.max():.2e}]")
        print(f"  Zero stress samples: {(stress_magnitude <= 1e-6).sum()}")
        print(f"  Valid samples: {valid_mask.sum()}/{len(df)}")
        
        if valid_mask.sum() != len(df):
            print(f"Filtering negligible stress states: {len(df)} -> {valid_mask.sum()} samples")
            base_features = base_features[valid_mask].reset_index(drop=True)
            stress_tensor = stress_tensor[valid_mask].reset_index(drop=True)
            df = df[valid_mask].reset_index(drop=True)
            stress_magnitude = stress_magnitude[valid_mask]
        
        # === Core decoupled prediction logic ===
        
        # 1. Calculate 1-dimensional failure length L as prediction target
        L_values = stress_magnitude  # L = ||σ||
        
        # 2. Calculate 6-dimensional stress direction vectors
        direction_vectors = np.zeros_like(stress_tensor.values)
        for i in range(len(stress_tensor)):
            if stress_magnitude.iloc[i] > 1e-8:
                direction_vectors[i] = stress_tensor.iloc[i].values / stress_magnitude.iloc[i]
            else:
                direction_vectors[i] = np.zeros(6)  # Zero vector for zero stress
        
        # 3. Construct 20-dimensional enhanced features: 14-dim material features + 6-dim stress direction
        enhanced_features = np.hstack([
            base_features.values,     # 14-dimensional material features
            direction_vectors         # 6-dimensional stress direction
        ])
        
        # Convert to DataFrame for easier processing
        direction_column_names = [f'dir_{col}' for col in stress_columns]
        enhanced_feature_names = base_features.columns.tolist() + direction_column_names
        enhanced_features_df = pd.DataFrame(enhanced_features, columns=enhanced_feature_names)
        L_series = pd.Series(L_values, name='failure_length_L')
        
        print(f"\n🎯 Decoupled prediction data construction completed:")
        print(f"  Enhanced features (20-dim): 14 material properties + 6 stress direction components")
        print(f"  Prediction target (1-dim): Failure length L = ||σ||")
        print(f"  L value range: [{L_series.min():.2f}, {L_series.max():.2f}] MPa")
        
        # Statistics of stress direction distribution
        print(f"\n📊 Stress direction vector statistics:")
        for i, col in enumerate(direction_column_names):
            values = direction_vectors[:, i]
            print(f"  {col}: min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")
        
        # Save case_id for Tsai-Wu coefficient lookup
        case_ids = df['case_id'].reset_index(drop=True) if 'case_id' in df.columns else None
        
        if case_ids is not None:
            print(f"\n📋 Case ID distribution: {case_ids.value_counts().sort_index().to_dict()}")
        
        self.feature_names = enhanced_feature_names
        
        # Save original stress tensor and direction vectors for reconstruction and physics loss calculation in training
        self.original_stress_tensor = stress_tensor.values  # Ensure original stress data is saved
        self.direction_vectors = direction_vectors
        
        print(f"\n✅ Dual-constraint feature preparation completed:")
        print(f"  Enhanced feature dimension: {enhanced_features_df.shape[1]}")
        print(f"  Prediction target dimension: 1 (failure length L)")
        print(f"  Valid samples: {enhanced_features_df.shape[0]}")
        print(f"  Original stress data saved: {self.original_stress_tensor.shape}")
        
        return enhanced_features_df, L_series, case_ids, direction_vectors, stress_tensor.values
    
    def fit_transform(self, enhanced_features, L_values):
        """Normalize data"""
        print("🔧 Dual-constraint mode: Applying Min-Max normalization to 20-dimensional features and 1-dimensional L values...")
        
        # Detailed input data check
        print(f"\n🔍 Pre-normalization data check:")
        print(f"  Enhanced features shape: {enhanced_features.shape}")
        print(f"  L value shape: {L_values.shape if hasattr(L_values, 'shape') else len(L_values)}")
        
        # Check L value distribution
        if hasattr(L_values, 'values'):
            L_array = L_values.values
        else:
            L_array = np.array(L_values)
        
        print(f"  L value statistics: min={L_array.min():.2f}, max={L_array.max():.2f}, mean={L_array.mean():.2f}, std={L_array.std():.2f}")
        
        # Check for anomalous L values
        zero_L_count = (L_array <= 1e-6).sum()
        large_L_count = (L_array > 10000).sum()
        print(f"  Extremely small L values (<=1e-6): {zero_L_count}, Extremely large L values (>10000): {large_L_count}")
        
        # Check feature ranges
        feature_ranges = enhanced_features.max() - enhanced_features.min()
        zero_range_features = feature_ranges[feature_ranges < 1e-10]
        if len(zero_range_features) > 0:
            print(f"⚠️ Warning: Found zero-range feature columns: {zero_range_features.index.tolist()}")
            # Add small random perturbation to zero-range features
            for col in zero_range_features.index:
                enhanced_features[col] += np.random.normal(0, 1e-8, len(enhanced_features))
        
        # Normalize 20-dimensional enhanced features
        print("📊 Enhanced feature normalization details:")
        features_scaled = self.feature_scaler.fit_transform(enhanced_features)
        for i, col in enumerate(enhanced_features.columns[:5]):  # Display first 5 columns
            orig_min, orig_max = enhanced_features[col].min(), enhanced_features[col].max()
            norm_min, norm_max = features_scaled[:, i].min(), features_scaled[:, i].max()
            print(f"  {col}: [{orig_min:.2e}, {orig_max:.2e}] -> [{norm_min:.4f}, {norm_max:.4f}]")
        
        print(f"  ... (showing first 5 columns, total {enhanced_features.shape[1]} columns)")
        
        # Normalize 1-dimensional L values - fix potential issues
        if self.use_L_normalization:
            print("\n📊 L value normalization details:")
            L_values_array = L_array.reshape(-1, 1)
            
            # Check if L value range is reasonable
            L_min, L_max = L_values_array.min(), L_values_array.max()
            L_range = L_max - L_min
            
            if L_range < 1e-6:
                print(f"⚠️ Warning: L value range too small ({L_range:.2e}), may affect training")
                # Add small perturbation to avoid division by zero
                L_values_array += np.random.normal(0, 1e-6, L_values_array.shape)
                L_min, L_max = L_values_array.min(), L_values_array.max()
            
            L_scaled = self.L_scaler.fit_transform(L_values_array).flatten()
            
            norm_min, norm_max = L_scaled.min(), L_scaled.max()
            norm_mean, norm_std = L_scaled.mean(), L_scaled.std()
            print(f"  L value: [{L_min:.2f}, {L_max:.2f}] MPa -> [{norm_min:.4f}, {norm_max:.4f}]")
            print(f"  After normalization: mean={norm_mean:.4f}, std={norm_std:.4f}")
            print("✅ L value Min-Max normalization enabled")
            
            # Check normalized distribution
            if norm_std < 0.1:
                print(f"⚠️ Warning: Normalized L value standard deviation too small ({norm_std:.4f}), may affect learning")
        else:
            L_scaled = L_array
            print("❌ Keep L values in original units")
        
        self.is_fitted = True
        
        print("\n✅ Dual-constraint data normalization completed")
        print(f"  Enhanced features normalized: 20 columns independently normalized to [0, 1]")
        print(f"  L value normalized: normalized to [0, 1]" if self.use_L_normalization else "  L value: original units")
        print(f"  Final feature shape: {features_scaled.shape}")
        print(f"  Final L value shape: {L_scaled.shape}")
        
        # Post-normalization data quality check
        print(f"\n🔍 Post-normalization data quality:")
        print(f"  NaN count in features: {np.isnan(features_scaled).sum()}")
        print(f"  Inf count in features: {np.isinf(features_scaled).sum()}")
        print(f"  NaN count in L values: {np.isnan(L_scaled).sum()}")
        print(f"  Inf count in L values: {np.isinf(L_scaled).sum()}")
        
        # Final data range check
        print(f"  Final feature range: [{features_scaled.min():.4f}, {features_scaled.max():.4f}]")
        print(f"  Final L value range: [{L_scaled.min():.4f}, {L_scaled.max():.4f}]")
        
        return features_scaled, L_scaled
    
    def transform(self, enhanced_features, L_values=None):
        """Transform data (already fitted)"""
        if not self.is_fitted:
            raise ValueError("Scalers not fitted. Call fit_transform first.")
        
        features_scaled = self.feature_scaler.transform(enhanced_features)
        
        if L_values is not None:
            if self.use_L_normalization:
                L_values_array = L_values.values.reshape(-1, 1) if hasattr(L_values, 'values') else np.array(L_values).reshape(-1, 1)
                L_scaled = self.L_scaler.transform(L_values_array).flatten()
            else:
                L_scaled = L_values.values if hasattr(L_values, 'values') else np.array(L_values)
            return features_scaled, L_scaled
        
        return features_scaled
    
    def inverse_transform_L(self, L_scaled):
        """Convert normalized L values back to original units"""
        if self.use_L_normalization and self.is_fitted:
            L_scaled_reshaped = L_scaled.reshape(-1, 1) if L_scaled.ndim == 1 else L_scaled
            return self.L_scaler.inverse_transform(L_scaled_reshaped).flatten()
        else:
            return L_scaled
    
    def inverse_transform_features(self, features_scaled):
        """Convert normalized enhanced features back to original units"""
        if self.is_fitted:
            return self.feature_scaler.inverse_transform(features_scaled)
        else:
            return features_scaled
    
    def reconstruct_stress_tensor(self, L_pred, direction_vectors):
        """Reconstruct stress tensor"""
        if isinstance(L_pred, (int, float)):
            L_pred = np.array([L_pred])
        
        if L_pred.ndim == 1 and direction_vectors.ndim == 2:
            # L_pred: (batch_size,), direction_vectors: (batch_size, 6)
            return L_pred.reshape(-1, 1) * direction_vectors
        elif L_pred.ndim == 2 and L_pred.shape[1] == 1:
            # L_pred: (batch_size, 1), direction_vectors: (batch_size, 6)
            return L_pred * direction_vectors
        else:
            raise ValueError(f"Shape mismatch: L_pred={L_pred.shape}, direction_vectors={direction_vectors.shape}")
    
    def split_by_case_and_merge(self, features, labels, case_ids, 
                               test_size=0.2, val_size=0.1, random_state=42):
        """Split dataset by case"""
        print(f"📊 Dual-constraint mode: Unified dataset splitting by case...")
        
        if case_ids is None:
            case_size = max(50, len(features) // 20)
            case_ids = pd.Series(range(len(features)) // case_size)
        
        unique_cases = case_ids.unique()
        train_indices, val_indices, test_indices = [], [], []
        
        for case in unique_cases:
            case_mask = case_ids == case
            case_indices = case_ids[case_mask].index.tolist()
            n_case_samples = len(case_indices)
            
            if n_case_samples < 5:
                train_indices.extend(case_indices)
                continue
            
            n_test = max(1, int(n_case_samples * test_size))
            n_val = max(1, int(n_case_samples * val_size))
            n_train = n_case_samples - n_test - n_val
            
            if n_train < 1:
                n_train = 1
                n_val = max(0, n_case_samples - n_train - n_test)
                n_test = n_case_samples - n_train - n_val
            
            np.random.seed(random_state)
            shuffled_indices = np.random.permutation(case_indices)
            
            case_train = shuffled_indices[:n_train].tolist()
            case_val = shuffled_indices[n_train:n_train+n_val].tolist()
            case_test = shuffled_indices[n_train+n_val:n_train+n_val+n_test].tolist()
            
            train_indices.extend(case_train)
            val_indices.extend(case_val)
            test_indices.extend(case_test)
        
        # Dataset splitting
        X_train = features.iloc[train_indices].reset_index(drop=True)
        X_val = features.iloc[val_indices].reset_index(drop=True) if val_indices else pd.DataFrame()
        X_test = features.iloc[test_indices].reset_index(drop=True)
        
        y_train = labels.iloc[train_indices].reset_index(drop=True) if hasattr(labels, 'iloc') else labels[train_indices]
        y_val = labels.iloc[val_indices].reset_index(drop=True) if val_indices and hasattr(labels, 'iloc') else pd.Series()
        y_test = labels.iloc[test_indices].reset_index(drop=True) if hasattr(labels, 'iloc') else labels[test_indices]
        
        case_train_ids = case_ids.iloc[train_indices].reset_index(drop=True)
        case_val_ids = case_ids.iloc[val_indices].reset_index(drop=True) if val_indices else pd.Series()
        case_test_ids = case_ids.iloc[test_indices].reset_index(drop=True)
        
        # Split direction vectors
        direction_train = self.direction_vectors[train_indices] if hasattr(self, 'direction_vectors') else None
        direction_val = self.direction_vectors[val_indices] if hasattr(self, 'direction_vectors') and val_indices else None
        direction_test = self.direction_vectors[test_indices] if hasattr(self, 'direction_vectors') else None
        
        case_info = {
            'unique_cases': [int(c) for c in unique_cases],
            'case_counts': {int(k): int(v) for k, v in case_ids.value_counts().to_dict().items()}
        }
        
        print(f"✅ Dual-constraint dataset splitting completed:")
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Validation set: {len(X_val)} samples")
        print(f"  Test set: {len(X_test)} samples")
        
        return (X_train, X_val, X_test, y_train, y_val, y_test, 
                case_train_ids, case_val_ids, case_test_ids, case_info,
                direction_train, direction_val, direction_test)
    
    # Backward-compatible interface - redirect to decoupled prediction method
    def prepare_features_and_labels(self, df):
        """Backward-compatible interface - redirect to decoupled prediction method"""
        enhanced_features, L_values, case_ids, direction_vectors, original_stress = self.prepare_decoupled_features_and_labels(df)
        return enhanced_features, L_values, case_ids
    
    @property
    def use_stress_normalization(self):
        """Backward-compatible property"""
        return self.use_L_normalization
    
    @use_stress_normalization.setter
    def use_stress_normalization(self, value):
        """Backward-compatible setter"""
        self.use_L_normalization = value
    
    @property
    def stress_scaler(self):
        """Return L value scaler as substitute for stress scaler"""
        return self.L_scaler

    def get_normalization_info(self):
        """Get normalization information"""
        if not self.is_fitted:
            return None
        
        info = {
            'feature_scaler': {
                'data_min_': self.feature_scaler.data_min_,
                'data_max_': self.feature_scaler.data_max_,
                'data_range_': self.feature_scaler.data_range_,
                'scale_': self.feature_scaler.scale_
            }
        }
        
        if self.use_L_normalization:
            info['L_scaler'] = {
                'data_min_': self.L_scaler.data_min_,
                'data_max_': self.L_scaler.data_max_, 
                'data_range_': self.L_scaler.data_range_,
                'scale_': self.L_scaler.scale_
            }
        
        return info

    def inverse_transform_stress(self, stress_scaled):
        """Convert normalized stress back to original units (backward-compatible interface)"""
        # In decoupled prediction mode, this method is mainly for backward compatibility
        # Actual stress reconstruction is done through the reconstruct_stress_tensor method
        return stress_scaled


# Backward-compatible alias
TsaiWuPINNDataProcessor = DualConstraintDataProcessor

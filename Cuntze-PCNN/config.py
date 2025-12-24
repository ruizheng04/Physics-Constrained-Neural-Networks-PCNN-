# config.py
# Responsibility: Store all static configurations and constants, including model hyperparameters, 
# training configurations, and physical constants.

# Cuntze Criterion Parameter Table - Extracted from puckcuntze.csv
# Key is case ID, value is the relevant strength and fracture parameters for that material
# Add Cuntze criterion parameters - ensure consistency with Tsai-Wu coefficient case
CUNTZE_PARAMETERS = {
    1: {"s2T": 100.0, "s2C": -150.0, "t12": 80.0, "t23": 60.0, "p_plus": 0.3, "p_minus": 0.25},
    2: {"s2T": 95.0, "s2C": -140.0, "t12": 75.0, "t23": 55.0, "p_plus": 0.3, "p_minus": 0.25},
    3: {"s2T": 90.0, "s2C": -130.0, "t12": 70.0, "t23": 50.0, "p_plus": 0.3, "p_minus": 0.25},
    4: {"s2T": 95.0, "s2C": -140.0, "t12": 75.0, "t23": 55.0, "p_plus": 0.3, "p_minus": 0.25},
    5: {"s2T": 95.0, "s2C": -140.0, "t12": 75.0, "t23": 55.0, "p_plus": 0.3, "p_minus": 0.25},
    6: {"s2T": 100.0, "s2C": -150.0, "t12": 80.0, "t23": 60.0, "p_plus": 0.3, "p_minus": 0.25},
    7: {"s2T": 90.0, "s2C": -130.0, "t12": 70.0, "t23": 50.0, "p_plus": 0.3, "p_minus": 0.25},
    8: {"s2T": 85.0, "s2C": -125.0, "t12": 65.0, "t23": 45.0, "p_plus": 0.3, "p_minus": 0.25},
    9: {"s2T": 105.0, "s2C": -160.0, "t12": 85.0, "t23": 65.0, "p_plus": 0.3, "p_minus": 0.25},
    10: {"s2T": 105.0, "s2C": -160.0, "t12": 85.0, "t23": 65.0, "p_plus": 0.3, "p_minus": 0.25},
    11: {"s2T": 90.0, "s2C": -130.0, "t12": 70.0, "t23": 50.0, "p_plus": 0.3, "p_minus": 0.25},
    12: {"s2T": 88.0, "s2C": -135.0, "t12": 72.0, "t23": 52.0, "p_plus": 0.3, "p_minus": 0.25},
    13: {"s2T": 92.0, "s2C": -145.0, "t12": 78.0, "t23": 58.0, "p_plus": 0.3, "p_minus": 0.25},
    14: {"s2T": 90.0, "s2C": -130.0, "t12": 70.0, "t23": 50.0, "p_plus": 0.3, "p_minus": 0.25},
    15: {"s2T": 98.0, "s2C": -155.0, "t12": 82.0, "t23": 62.0, "p_plus": 0.3, "p_minus": 0.25},
    16: {"s2T": 98.0, "s2C": -155.0, "t12": 82.0, "t23": 62.0, "p_plus": 0.3, "p_minus": 0.25}
}

# Fixed Configuration - Resolve poor training results + dual physical constraints
FIXED_CONFIG = {
    # Network architecture
    "layer_sizes": [20, 512, 1],
    "activation": "swiglu",
    "lr": 0.01,
    "dropout_rate": 0.01,
    "weight_decay": 0.0,
    "use_batch_norm": False,
    "use_residual": False,
    
    # Training settings
    "max_epochs": 3000,
    "early_stop_patience": 500,
    "warm_start_epochs": 800,
    "batch_size": 64,
    
    # Tsai-Wu Physical Loss Configuration
    "lambda_tsai_wu_final": 0.5,
    "tsai_wu_delay_epochs": 1500,
    "W_tsai_wu": 0.2,
    
    # Cuntze Physical Loss Configuration
    "lambda_cuntze_final": 0.5,  # Final weight for Cuntze criterion
    "cuntze_delay_epochs": 1600,  # Delay epochs for Cuntze criterion introduction (can differ from Tsai-Wu)
    "W_cuntze": 0.2,            # Static weight for Cuntze criterion
    
    # Backward compatibility: old parameter mapping
    "lambda_final": 0.5,  # Maps to lambda_tsai_wu_final
    "physics_delay_epochs": 1500,  # Maps to tsai_wu_delay_epochs
    "W_physics": 0.2,  # Maps to W_tsai_wu
    
    # Other settings
    "adaptive_balance": False,
    "gradient_balance_alpha": 0.1,
    "physics_warmup_rate": 0.01,
    "normalization_method": "minmax_columnwise",
    "l1_lambda": 0.0,
    "l2_lambda": 0.0,
    "verbose": True,
    
    # SA-PINN settings (disabled by default)
    "use_sa_pinn": False,
    "sa_pinn_lr": 0.001,
    "weight_update_freq": 1,
    "weight_clip_max": 10.0,
    "weight_init_value": 1.0,
}

# Decoupled Prediction Test Configuration - Further simplified for debugging
TEST_CONFIG = {
    "layer_sizes": [20, 16, 1],  # Minimal network structure
    "activation": "swiglu",  # Activation function for hidden layers (last layer forces Sigmoid)
    "lr": 0.001,  # Conservative learning rate
    "dropout_rate": 0.0,  # Completely remove dropout
    "weight_decay": 0.0,  # Completely remove regularization
    "use_batch_norm": False,
    "use_residual": False,
    "lambda_final": 0.0,  # Completely disable physical constraints
    "physics_delay_epochs": 5000,  # Delay until end of training
    "warm_start_epochs": 3000,  # Long pure data training phase
    "W_physics": 0.0,  # Completely set to 0
    "adaptive_balance": False,
    "gradient_balance_alpha": 0.1,
    "physics_warmup_rate": 0.01,
    "normalization_method": "minmax_columnwise",
    "max_epochs": 100,
    "early_stop_patience": 50,
}

# Tsai-Wu Coefficient Table - Precomputed
TSAI_WU_COEFFICIENTS = {
    1: {"F1": -1.62e-04, "F2": 1.58e-02, "F3": 1.58e-02, "F11": 3.47e-07, "F22": 1.04e-04, "F33": 1.04e-04, 
        "F44": 1.59e-04, "F55": 1.59e-04, "F66": 1.59e-04, "F12": -3.01e-06, "F13": -3.01e-06, "F23": -5.21e-05},
    2: {"F1": -8.77e-04, "F2": 2.00e-02, "F3": 2.00e-02, "F11": 1.54e-06, "F22": 2.51e-04, "F33": 2.51e-04,
        "F44": 1.93e-04, "F55": 1.93e-04, "F66": 1.93e-04, "F12": -9.82e-06, "F13": -9.82e-06, "F23": -1.25e-04},
    3: {"F1": -2.38e-04, "F2": 8.29e-03, "F3": 1.05e-02, "F11": 2.46e-07, "F22": 7.40e-05, "F33": 8.59e-05,
        "F44": 1.23e-04, "F55": 3.08e-04, "F66": 1.23e-04, "F12": -2.13e-06, "F13": -2.30e-06, "F23": -3.99e-05},
    4: {"F1": -8.77e-04, "F2": 2.00e-02, "F3": 2.00e-02, "F11": 1.54e-06, "F22": 2.51e-04, "F33": 2.51e-04,
        "F44": 1.93e-04, "F55": 1.93e-04, "F66": 1.93e-04, "F12": -9.82e-06, "F13": -9.82e-06, "F23": -1.25e-04},
    5: {"F1": -8.77e-04, "F2": 2.00e-02, "F3": 2.00e-02, "F11": 1.54e-06, "F22": 2.51e-04, "F33": 2.51e-04,
        "F44": 1.93e-04, "F55": 1.93e-04, "F66": 1.93e-04, "F12": -9.82e-06, "F13": -9.82e-06, "F23": -1.25e-04},
    6: {"F1": -1.62e-04, "F2": 1.58e-02, "F3": 1.58e-02, "F11": 3.47e-07, "F22": 1.04e-04, "F33": 1.04e-04,
        "F44": 1.59e-04, "F55": 1.59e-04, "F66": 1.59e-04, "F12": -3.01e-06, "F13": -3.01e-06, "F23": -5.21e-05},
    7: {"F1": -2.38e-04, "F2": 8.29e-03, "F3": 1.05e-02, "F11": 2.46e-07, "F22": 7.40e-05, "F33": 8.59e-05,
        "F44": 1.23e-04, "F55": 3.08e-04, "F66": 1.23e-04, "F12": -2.13e-06, "F13": -2.30e-06, "F23": -3.99e-05},
    8: {"F1": -8.33e-04, "F2": 1.67e-02, "F3": 1.67e-02, "F11": 1.39e-06, "F22": 2.08e-04, "F33": 2.08e-04,
        "F44": 1.78e-04, "F55": 1.78e-04, "F66": 1.78e-04, "F12": -8.51e-06, "F13": -8.51e-06, "F23": -1.04e-04},
    9: {"F1": 0.00e+00, "F2": 2.01e-02, "F3": 2.01e-02, "F11": 4.97e-07, "F22": 1.05e-04, "F33": 1.05e-04,
        "F44": 1.38e-04, "F55": 1.38e-04, "F66": 1.38e-04, "F12": -3.61e-06, "F13": -3.61e-06, "F23": -5.25e-05},
    10: {"F1": 0.00e+00, "F2": 2.01e-02, "F3": 2.01e-02, "F11": 4.97e-07, "F22": 1.05e-04, "F33": 1.05e-04,
         "F44": 1.38e-04, "F55": 1.38e-04, "F66": 1.38e-04, "F12": -3.61e-06, "F13": -3.61e-06, "F23": -5.25e-05},
    11: {"F1": -2.38e-04, "F2": 8.29e-03, "F3": 1.05e-02, "F11": 2.46e-07, "F22": 7.40e-05, "F33": 8.59e-05,
         "F44": 1.23e-04, "F55": 3.08e-04, "F66": 1.23e-04, "F12": -2.13e-06, "F13": -2.30e-06, "F23": -3.99e-05},
    12: {"F1": -4.46e-04, "F2": 1.89e-02, "F3": 1.89e-02, "F11": 4.18e-07, "F22": 1.96e-04, "F33": 1.96e-04,
         "F44": 1.78e-04, "F55": 4.34e-04, "F66": 1.78e-04, "F12": -4.53e-06, "F13": -4.53e-06, "F23": -9.82e-05},
    13: {"F1": -2.15e-04, "F2": 1.30e-02, "F3": 1.30e-02, "F11": 2.96e-07, "F22": 7.89e-05, "F33": 7.89e-05,
         "F44": 1.06e-04, "F55": 1.06e-04, "F66": 1.06e-04, "F12": -2.42e-06, "F13": -2.42e-06, "F23": -3.95e-05},
    14: {"F1": -2.38e-04, "F2": 8.29e-03, "F3": 1.05e-02, "F11": 2.46e-07, "F22": 7.40e-05, "F33": 8.59e-05,
         "F44": 1.23e-04, "F55": 3.08e-04, "F66": 1.23e-04, "F12": -2.13e-06, "F13": -2.30e-06, "F23": -3.99e-05},
    15: {"F1": -1.48e-04, "F2": 9.53e-03, "F3": 9.53e-03, "F11": 2.51e-07, "F22": 4.76e-05, "F33": 4.76e-05,
         "F44": 1.23e-04, "F55": 1.23e-04, "F66": 1.23e-04, "F12": -1.73e-06, "F13": -1.73e-06, "F23": -2.38e-05},
    16: {"F1": -1.48e-04, "F2": 9.53e-03, "F3": 9.53e-03, "F11": 2.51e-07, "F22": 4.76e-05, "F33": 4.76e-05,
         "F44": 1.23e-04, "F55": 1.23e-04, "F66": 1.23e-04, "F12": -1.73e-06, "F13": -1.73e-06, "F23": -2.38e-05}
}

# Fix TSAI_WU_COEFFICIENTS - Use meaningful non-zero values
TSAI_WU_COEFFICIENTS = {
    1: {
        # Linear terms (moderate values)
        'F1': 0.01, 'F2': 0.01, 'F3': 0.01,
        # Quadratic terms (significantly increased, ensure loss generation)
        'F11': 0.001, 'F22': 0.001, 'F33': 0.001,
        'F44': 0.001, 'F55': 0.001, 'F66': 0.001,
        # Interaction terms (small but non-zero)
        'F12': 0.0001, 'F13': 0.0001, 'F23': 0.0001
    }
}

# Fix CUNTZE_PARAMETERS - Use more easily triggerable thresholds
CUNTZE_PARAMETERS = {
    1: {
        'R_par_t': 50.0,   # Lower threshold for easier triggering
        'R_par_s': 25.0,   # Lower threshold for easier triggering
        's2T': 50.0,
        's2C': 200.0,
        't12': 70.0,
        't23': 30.0,
        'p_plus': 0.25,
        'p_minus': 0.25
    }
}

# SA-PINN Configuration ensure completeness
SA_PINN_CONFIG = {
    'use_sa_pinn': True,
    'sa_pinn_lr': 0.001,
    'weight_update_freq': 1,
    'weight_clip_max': 10.0,
    'weight_init_value': 1.0,
    'use_normalization': True,
}

# Update OPTIMAL_CONFIG
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
    'warm_start_epochs': 600,
    
    # Dual Physical Constraint Weights - optimized configuration
    'lambda_tsai_wu_final': 0.5,
    'tsai_wu_delay_epochs': 1500,
    'W_tsai_wu': 0.2,
    
    'lambda_cuntze_final': 0.5,
    'cuntze_delay_epochs': 1600,
    'W_cuntze': 0.2,
    
    # Backward compatibility
    'lambda_final': 0.5,
    'physics_delay_epochs': 1500,
    'W_physics': 0.2,
    
    # SA-PINN Adaptive Weight Configuration
    **SA_PINN_CONFIG,
    
    # Physical constraint settings
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

# Ensure FIXED_CONFIG contains necessary configuration
if 'FIXED_CONFIG' not in globals():
    FIXED_CONFIG = OPTIMAL_CONFIG.copy()
    FIXED_CONFIG['use_sa_pinn'] = False  # Standard mode default does not enable SA-PINN

# Use aggressive coefficients to ensure physical loss generation
TSAI_WU_COEFFICIENTS = {
    1: {
        # Linear terms (significantly increased)
        'F1': 0.1, 'F2': 0.1, 'F3': 0.1,
        # Quadratic terms (significantly increased)
        'F11': 0.01, 'F22': 0.01, 'F33': 0.01,
        'F44': 0.01, 'F55': 0.01, 'F66': 0.01,
        # Interaction terms (increased coupling)
        'F12': 0.001, 'F13': 0.001, 'F23': 0.001
    }
}

# Use extremely low thresholds to ensure failure triggering
CUNTZE_PARAMETERS = {
    1: {
        'R_par_t': 10.0,   # Significantly lower threshold
        'R_par_s': 5.0,    # Significantly lower threshold
        's2T': 50.0,
        's2C': 200.0,
        't12': 70.0,
        't23': 30.0,
        'p_plus': 0.25,
        'p_minus': 0.25
    }
}
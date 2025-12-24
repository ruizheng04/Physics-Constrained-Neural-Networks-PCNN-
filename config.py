# config.py
# Responsibility: Store all static configurations and constants, including model hyperparameters, training configuration, and physical constants.

# Fixed configuration - to solve poor training effectiveness
FIXED_CONFIG = {
    "layer_sizes": [20, 512, 1],  # Reduce network capacity, improve stability
    "activation": "swiglu",  # Hidden layer activation (last layer forced to use Sigmoid)
    "lr": 0.01,  # Moderate learning rate
    "dropout_rate": 0.01,  # Very low dropout
    "weight_decay": 0.0001,  # Reduce regularization
    "use_batch_norm": False,
    "use_residual": False,
    "lambda_final": 0.5,  # Moderate physics constraint weight
    "physics_delay_epochs": 1500,  # Delay physics loss intervention
    "warm_start_epochs": 800,  # Pure data training phase
    "W_physics": 0.2,  # Moderate physics weight
    "adaptive_balance": False,
    "gradient_balance_alpha": 0.1,
    "physics_warmup_rate": 0.01,
    "normalization_method": "minmax_columnwise",
}

# Decoupled prediction test configuration - further simplified for debugging
TEST_CONFIG = {
    "layer_sizes": [20, 16, 1],  # Simplest network structure
    "activation": "swiglu",  # Hidden layer activation (last layer forced to use Sigmoid)
    "lr": 0.001,  # Conservative learning rate
    "dropout_rate": 0.0,  # Completely remove dropout
    "weight_decay": 0.0,  # Completely remove regularization
    "use_batch_norm": False,
    "use_residual": False,
    "lambda_final": 0.0,  # Completely disable physics constraint
    "physics_delay_epochs": 5000,  # Postpone to end of training
    "warm_start_epochs": 3000,  # Long pure data training
    "W_physics": 0.0,  # Completely set to 0
    "adaptive_balance": False,
    "gradient_balance_alpha": 0.1,
    "physics_warmup_rate": 0.01,
    "normalization_method": "minmax_columnwise",
}

# Tsai-Wu coefficient table - pre-calculated
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

# LaRC准则材料参数 - 根据datasetnew.csv中的实际材料强度调整
LARC_MATERIAL_PROPERTIES = {
    # Case 1: High-strength carbon fiber composite
    1: {
        "X_T": 1950.0,   # Use strength_1 from dataset
        "X_C": 1480.0,   # Use strength_2 from dataset
        "Y_T": 48.0,     # Use strength_3 from dataset
        "Y_C": 200.0,    # Use strength_4 from dataset
        "S_L": 79.0,     # Use strength_7 from dataset (in-plane shear)
        "S_T": 79.0,     # Use strength_8 from dataset (out-of-plane shear)
        "S_12": 79.0,    # Use strength_9 from dataset
        "G_12": 4500.0,  # Estimated shear modulus
        "eta_T": 0.3,    # Transverse friction coefficient
        "eta_L": 0.3,    # Longitudinal friction coefficient
    },
    
    # Case 2: Medium-strength glass fiber composite
    2: {
        "X_T": 1140.0, "X_C": 570.0, "Y_T": 35.0, "Y_C": 114.0,
        "S_L": 72.0, "S_T": 72.0, "S_12": 72.0, "G_12": 3500.0,
        "eta_T": 0.3, "eta_L": 0.3,
    },
    
    # Case 3: High-performance carbon fiber composite
    3: {
        "X_T": 2560.0, "X_C": 1590.0, "Y_T": 73.0, "Y_C": 185.0,
        "S_L": 90.0, "S_T": 90.0, "S_12": 57.0, "G_12": 5500.0,
        "eta_T": 0.3, "eta_L": 0.3,
    },
    
    # Case 4: Same material system as Case 2
    4: {
        "X_T": 1140.0, "X_C": 570.0, "Y_T": 35.0, "Y_C": 114.0,
        "S_L": 72.0, "S_T": 72.0, "S_12": 72.0, "G_12": 3500.0,
        "eta_T": 0.3, "eta_L": 0.3,
    },
    
    # Case 5: Same as Case 4
    5: {
        "X_T": 1140.0, "X_C": 570.0, "Y_T": 35.0, "Y_C": 114.0,
        "S_L": 72.0, "S_T": 72.0, "S_12": 72.0, "G_12": 3500.0,
        "eta_T": 0.3, "eta_L": 0.3,
    },
    
    # Case 6: Same high-strength material as Case 1
    6: {
        "X_T": 1950.0, "X_C": 1480.0, "Y_T": 48.0, "Y_C": 200.0,
        "S_L": 79.0, "S_T": 79.0, "S_12": 79.0, "G_12": 4500.0,
        "eta_T": 0.3, "eta_L": 0.3,
    },
    
    # Case 7: Same as Case 3
    7: {
        "X_T": 2560.0, "X_C": 1590.0, "Y_T": 73.0, "Y_C": 185.0,
        "S_L": 90.0, "S_T": 90.0, "S_12": 57.0, "G_12": 5500.0,
        "eta_T": 0.3, "eta_L": 0.3,
    },
    
    # Case 8: Medium-strength material
    8: {
        "X_T": 1200.0, "X_C": 600.0, "Y_T": 40.0, "Y_C": 120.0,
        "S_L": 75.0, "S_T": 75.0, "S_12": 75.0, "G_12": 3800.0,
        "eta_T": 0.3, "eta_L": 0.3,
    },
    
    # Case 9: High-strength isotropic material
    9: {
        "X_T": 1420.0, "X_C": 1420.0, "Y_T": 41.0, "Y_C": 232.0,
        "S_L": 85.0, "S_T": 85.0, "S_12": 85.0, "G_12": 4200.0,
        "eta_T": 0.3, "eta_L": 0.3,
    },
    
    # Case 10: Same as Case 9
    10: {
        "X_T": 1420.0, "X_C": 1420.0, "Y_T": 41.0, "Y_C": 232.0,
        "S_L": 85.0, "S_T": 85.0, "S_12": 85.0, "G_12": 4200.0,
        "eta_T": 0.3, "eta_L": 0.3,
    },
    
    # Case 11: Same as Case 7
    11: {
        "X_T": 2560.0, "X_C": 1590.0, "Y_T": 73.0, "Y_C": 185.0,
        "S_L": 90.0, "S_T": 90.0, "S_12": 57.0, "G_12": 5500.0,
        "eta_T": 0.3, "eta_L": 0.3,
    },
    
    # Case 12: High-strength material
    12: {
        "X_T": 2172.0, "X_C": 1103.0, "Y_T": 38.0, "Y_C": 134.0,
        "S_L": 75.0, "S_T": 75.0, "S_12": 48.0, "G_12": 4800.0,
        "eta_T": 0.3, "eta_L": 0.3,
    },
    
    # Case 13: Ultra-high-strength material
    13: {
        "X_T": 2234.0, "X_C": 1510.0, "Y_T": 57.0, "Y_C": 222.0,
        "S_L": 97.0, "S_T": 97.0, "S_12": 97.0, "G_12": 5200.0,
        "eta_T": 0.3, "eta_L": 0.3,
    },
    
    # Case 14: Same as Case 7
    14: {
        "X_T": 2560.0, "X_C": 1590.0, "Y_T": 73.0, "Y_C": 185.0,
        "S_L": 90.0, "S_T": 90.0, "S_12": 57.0, "G_12": 5500.0,
        "eta_T": 0.3, "eta_L": 0.3,
    },
    
    # Case 15: Ultra-high-strength material
    15: {
        "X_T": 2310.0, "X_C": 1724.0, "Y_T": 76.0, "Y_C": 276.0,
        "S_L": 90.0, "S_T": 90.0, "S_12": 90.0, "G_12": 5400.0,
        "eta_T": 0.3, "eta_L": 0.3,
    },
    
    # Case 16: Same as Case 15
    16: {
        "X_T": 2310.0, "X_C": 1724.0, "Y_T": 76.0, "Y_C": 276.0,
        "S_L": 90.0, "S_T": 90.0, "S_12": 90.0, "G_12": 5400.0,
        "eta_T": 0.3, "eta_L": 0.3,
    },
}
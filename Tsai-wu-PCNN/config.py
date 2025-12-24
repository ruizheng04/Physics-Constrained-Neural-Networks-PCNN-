# config.py
# Responsibility: Store all static configurations and constants, including model hyperparameters, training configs, and physical constants.

# Fixed configuration - addressing poor training performance
FIXED_CONFIG = {
    "layer_sizes": [20,512, 1],  # Reduce network capacity for stability
    "activation": "swiglu",  # Hidden layer activation (last layer forced to use Sigmoid)
    "lr": 0.01,  # Moderate learning rate
    "dropout_rate": 0.01,  # Very low dropout
    "weight_decay": 0.0,  # Reduce regularization
    "use_batch_norm": False,
    "use_residual": False,
    "lambda_final": 0.5,  # Moderate physics constraint weight
    "physics_delay_epochs": 1500,  # Delay physics loss introduction
    "warm_start_epochs": 800,  # Pure data training phase
    "W_physics": 0.2,  # Moderate physics weight
    "adaptive_balance": False,
    "gradient_balance_alpha": 0.1,
    "physics_warmup_rate": 0.01,
    "normalization_method": "minmax_columnwise",
}

# Decoupled prediction test configuration - further simplified for debugging
TEST_CONFIG = {
    "layer_sizes": [20, 16, 1],  # Minimal network structure
    "activation": "swiglu",  # Hidden layer activation (last layer forced to use Sigmoid)
    "lr": 0.001,  # Conservative learning rate
    "dropout_rate": 0.0,  # Remove dropout completely
    "weight_decay": 0.0,  # Remove regularization completely
    "use_batch_norm": False,
    "use_residual": False,
    "lambda_final": 0.0,  # Disable physics constraints completely
    "physics_delay_epochs": 5000,  # Postpone to end of training
    "warm_start_epochs": 3000,  # Long pure data training
    "W_physics": 0.0,  # Set to 0 completely
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
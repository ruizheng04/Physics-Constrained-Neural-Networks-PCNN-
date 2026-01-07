# Adaptive Lambda Adjuster
# Dynamically adjust lambda weight based on physics loss contribution

import torch
import numpy as np

class AdaptiveLambdaAdjuster:
    """Adaptive Lambda Adjuster, dynamically adjusts weight based on physics loss contribution"""
    
    def __init__(self, initial_lambda=1.0, target_contribution=8.0, adjustment_rate=0.1):
        """
        initial_lambda: Initial lambda value
        target_contribution: Target physics loss contribution percentage
        adjustment_rate: Adjustment speed
        """
        self.current_lambda = initial_lambda
        self.target_contribution = target_contribution
        self.adjustment_rate = adjustment_rate
        
        # History records
        self.contribution_history = []
        self.lambda_history = []
        self.adjustment_count = 0
        
    def update_lambda(self, physics_contribution, epoch):
        """Update lambda based on current physics loss contribution"""
        
        # Record history
        self.contribution_history.append(physics_contribution)
        self.lambda_history.append(self.current_lambda)
        
        # Keep history of last 50 epochs
        if len(self.contribution_history) > 50:
            self.contribution_history.pop(0)
            self.lambda_history.pop(0)
        
        # Start adjusting after training progresses for a while
        if epoch < 100 or len(self.contribution_history) < 10:
            return self.current_lambda
        
        # Calculate recent average contribution
        recent_contribution = np.mean(self.contribution_history[-10:])
        
        # Calculate adjustment factor
        if recent_contribution < self.target_contribution * 0.3:  # Contribution too low
            # Aggressive adjustment
            adjustment_factor = 1.0 + self.adjustment_rate * 3
            self.current_lambda *= adjustment_factor
            self.adjustment_count += 1
            
        elif recent_contribution < self.target_contribution * 0.6:  # Contribution slightly low
            # Moderate adjustment
            adjustment_factor = 1.0 + self.adjustment_rate * 1.5
            self.current_lambda *= adjustment_factor
            self.adjustment_count += 1
            
        elif recent_contribution > self.target_contribution * 3:  # Contribution too high
            # Reduce weight
            adjustment_factor = 1.0 - self.adjustment_rate * 0.5
            self.current_lambda *= adjustment_factor
            self.adjustment_count += 1
        
        # Limit lambda within reasonable range
        self.current_lambda = np.clip(self.current_lambda, 0.01, 20.0)
        
        return self.current_lambda
    
    def get_adjustment_summary(self):
        """Get adjustment summary"""
        if len(self.contribution_history) < 5:
            return "Insufficient data"
        
        recent_contribution = np.mean(self.contribution_history[-5:])
        lambda_change = (self.current_lambda / self.lambda_history[0] - 1) * 100
        
        return {
            'current_lambda': self.current_lambda,
            'recent_contribution': recent_contribution,
            'target_contribution': self.target_contribution,
            'lambda_change_percent': lambda_change,
            'adjustment_count': self.adjustment_count
        }

# Usage example function
def create_adaptive_lambda_adjuster(config):
    """Create adaptive lambda adjuster based on configuration"""
    return AdaptiveLambdaAdjuster(
        initial_lambda=config.get('lambda_final', 1.0),
        target_contribution=8.0,  # Target 8% contribution
        adjustment_rate=0.15  # Adjustment speed
    )
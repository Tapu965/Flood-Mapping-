# This file makes inference a package
# Import both predictors so they're available
try:
    from .predictor import FloodPredictor
except:
    pass  # Original predictor might not exist

from .enhanced_predictor import EnhancedFloodPredictor

__all__ = ['FloodPredictor', 'EnhancedFloodPredictor']
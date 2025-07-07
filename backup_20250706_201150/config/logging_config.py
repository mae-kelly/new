import logging
import sys
from datetime import datetime

def setup_logging():
    """Setup comprehensive logging"""
    
    # Create logs directory
    import os
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # File handler
    file_handler = logging.FileHandler(
        f'logs/trading_bot_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

# Trading-specific loggers
def get_trading_logger():
    return logging.getLogger('trading')

def get_ml_logger():
    return logging.getLogger('ml')

def get_contract_logger():
    return logging.getLogger('contract_analysis')

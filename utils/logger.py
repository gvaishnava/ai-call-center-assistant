import logging
import sys

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Creates and configures a centralized logger for the application.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if the logger already exists
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Define formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
    return logger

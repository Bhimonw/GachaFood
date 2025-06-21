import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = 'INFO',
    format_string: Optional[str] = None,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """Setup logger with file and console handlers"""
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Create formatter
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is specified)
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get existing logger or create a basic one"""
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set up basic configuration
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class AppLogger:
    """Application logger manager"""
    
    def __init__(self, app_name: str = 'GachaFood', log_dir: str = 'logs'):
        self.app_name = app_name
        self.log_dir = log_dir
        self.loggers = {}
        
        # Ensure log directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    def get_logger(self, module_name: str, level: str = 'INFO') -> logging.Logger:
        """Get or create logger for specific module"""
        logger_name = f"{self.app_name}.{module_name}"
        
        if logger_name not in self.loggers:
            log_file = os.path.join(self.log_dir, f"{module_name}.log")
            self.loggers[logger_name] = setup_logger(
                name=logger_name,
                log_file=log_file,
                level=level
            )
        
        return self.loggers[logger_name]
    
    def get_main_logger(self, level: str = 'INFO') -> logging.Logger:
        """Get main application logger"""
        if 'main' not in self.loggers:
            log_file = os.path.join(self.log_dir, 'app.log')
            self.loggers['main'] = setup_logger(
                name=self.app_name,
                log_file=log_file,
                level=level
            )
        
        return self.loggers['main']
    
    def get_api_logger(self, level: str = 'INFO') -> logging.Logger:
        """Get API specific logger"""
        return self.get_logger('api', level)
    
    def get_ml_logger(self, level: str = 'INFO') -> logging.Logger:
        """Get ML/clustering specific logger"""
        return self.get_logger('ml', level)
    
    def get_data_logger(self, level: str = 'INFO') -> logging.Logger:
        """Get data processing specific logger"""
        return self.get_logger('data', level)
    
    def set_level_all(self, level: str) -> None:
        """Set log level for all loggers"""
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        for logger in self.loggers.values():
            logger.setLevel(log_level)
            for handler in logger.handlers:
                handler.setLevel(log_level)
    
    def add_file_handler_all(self, log_file: str, level: str = 'INFO') -> None:
        """Add file handler to all existing loggers"""
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        for logger in self.loggers.values():
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10485760,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

class LoggerMixin:
    """Mixin class to add logging capability to any class"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        if not hasattr(self, '_logger'):
            class_name = self.__class__.__name__
            self._logger = get_logger(f"GachaFood.{class_name}")
        return self._logger
    
    def log_info(self, message: str, *args, **kwargs) -> None:
        """Log info message"""
        self.logger.info(message, *args, **kwargs)
    
    def log_warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message"""
        self.logger.warning(message, *args, **kwargs)
    
    def log_error(self, message: str, *args, **kwargs) -> None:
        """Log error message"""
        self.logger.error(message, *args, **kwargs)
    
    def log_debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message"""
        self.logger.debug(message, *args, **kwargs)
    
    def log_exception(self, message: str, *args, **kwargs) -> None:
        """Log exception with traceback"""
        self.logger.exception(message, *args, **kwargs)

# Global app logger instance
app_logger = AppLogger()

# Convenience functions
def get_main_logger() -> logging.Logger:
    """Get main application logger"""
    return app_logger.get_main_logger()

def get_api_logger() -> logging.Logger:
    """Get API logger"""
    return app_logger.get_api_logger()

def get_ml_logger() -> logging.Logger:
    """Get ML logger"""
    return app_logger.get_ml_logger()

def get_data_logger() -> logging.Logger:
    """Get data logger"""
    return app_logger.get_data_logger()
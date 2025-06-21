import os
from typing import Dict, Any

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = False
    TESTING = False
    
    # Application settings
    APP_NAME = 'GachaFood'
    APP_VERSION = '1.0.0'
    
    # Data settings
    DATA_FILE_PATH = os.environ.get('DATA_FILE_PATH') or 'data/tempat_makan_cleaned.csv'
    
    # ML Model settings
    MODEL_SAVE_PATH = os.environ.get('MODEL_SAVE_PATH') or 'models/saved_models'
    CLUSTERING_ALGORITHM = 'kmeans'
    DEFAULT_N_CLUSTERS = 3  # Force 3 clusters to merge sedang and menengah
    AUTO_OPTIMIZE_CLUSTERS = False  # Disable auto-optimization
    MAX_CLUSTERS = 3
    RANDOM_STATE = 42
    
    # API settings
    API_PREFIX = '/api'
    MAX_RESULTS_PER_REQUEST = 50
    DEFAULT_RESULTS_LIMIT = 10
    
    # Pagination settings
    RESTAURANTS_PER_PAGE = 20
    MAX_RESTAURANTS_PER_PAGE = 100
    
    # Cache settings
    CACHE_TIMEOUT = 300  # 5 minutes
    
    # Logging settings
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Feature weights for hybrid scoring
    HYBRID_WEIGHTS = {
        'rating': 0.4,
        'distance': 0.35,
        'price': 0.25
    }
    
    # Clustering features and weights
    CLUSTERING_FEATURES = ['harga', 'rating', 'jarak', 'tipe_encoded']
    FEATURE_WEIGHTS = {
        'harga': 0.5,      # Increase price weight for better separation
        'rating': 0.3,     # Increase rating weight
        'jarak': 0.15,     # Reduce distance weight
        'tipe_encoded': 0.05 # Reduce place type weight
    }
    
    # Data validation rules
    DATA_VALIDATION = {
        'harga': {'min': 0, 'max': 1000000},  # Price range in IDR
        'rating': {'min': 0, 'max': 5},       # Rating range
        'jarak': {'min': 0, 'max': 100},      # Distance range in km
    }
    
    # Column mappings for CSV
    COLUMN_MAPPING = {
        'Nama Tempat': 'nama_tempat',
        'Tipe Tempat': 'tipe_tempat',
        'Estimasi Harga (RP.)': 'harga',
        'Rating Tempat Makan': 'rating',
        'Lokasi (maps)': 'lokasi',
        'Jarak dari kampus (km)': 'jarak'
    }
    
    @staticmethod
    def init_app(app):
        """Initialize application with config"""
        pass

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    
    # Development specific settings
    HOST = '127.0.0.1'
    PORT = 8080
    
    # More verbose logging in development
    LOG_LEVEL = 'DEBUG'
    
    # Shorter cache timeout for development
    CACHE_TIMEOUT = 60

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    
    # Production specific settings
    HOST = '0.0.0.0'
    PORT = int(os.environ.get('PORT', 5000))
    
    # Production logging
    LOG_LEVEL = 'WARNING'
    
    # Longer cache timeout for production
    CACHE_TIMEOUT = 600  # 10 minutes
    
    @staticmethod
    def init_app(app):
        Config.init_app(app)
        
        # Production specific initialization
        import logging
        from logging.handlers import RotatingFileHandler
        
        if not app.debug and not app.testing:
            # Setup file logging
            if not os.path.exists('logs'):
                os.mkdir('logs')
            
            file_handler = RotatingFileHandler(
                'logs/gachafood.log',
                maxBytes=10240000,  # 10MB
                backupCount=10
            )
            
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
            app.logger.setLevel(logging.INFO)
            app.logger.info('GachaFood startup')

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    
    # Testing specific settings
    DATA_FILE_PATH = 'tests/test_data.csv'
    CACHE_TIMEOUT = 0  # No caching in tests
    
    # Use in-memory database for testing
    WTF_CSRF_ENABLED = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name: str = None) -> Config:
    """Get configuration based on environment"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    return config.get(config_name, config['default'])

class AppSettings:
    """Application settings manager"""
    
    def __init__(self, config_obj: Config):
        self.config = config_obj
    
    def get_data_file_path(self) -> str:
        """Get absolute path to data file"""
        if os.path.isabs(self.config.DATA_FILE_PATH):
            return self.config.DATA_FILE_PATH
        else:
            # Relative to application root
            return os.path.join(os.getcwd(), self.config.DATA_FILE_PATH)
    
    def get_model_save_path(self) -> str:
        """Get absolute path to model save directory"""
        if os.path.isabs(self.config.MODEL_SAVE_PATH):
            return self.config.MODEL_SAVE_PATH
        else:
            return os.path.join(os.getcwd(), self.config.MODEL_SAVE_PATH)
    
    def ensure_directories(self) -> None:
        """Ensure required directories exist"""
        model_dir = self.get_model_save_path()
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        
        # Ensure logs directory for production
        if isinstance(self.config, ProductionConfig):
            logs_dir = 'logs'
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir, exist_ok=True)
    
    def validate_data_file(self) -> bool:
        """Validate that data file exists and is readable"""
        data_path = self.get_data_file_path()
        return os.path.exists(data_path) and os.path.isfile(data_path)
    
    def get_clustering_config(self) -> Dict[str, Any]:
        """Get clustering configuration"""
        return {
            'algorithm': self.config.CLUSTERING_ALGORITHM,
            'n_clusters': self.config.DEFAULT_N_CLUSTERS,
            'auto_optimize': getattr(self.config, 'AUTO_OPTIMIZE_CLUSTERS', True),
            'max_clusters': getattr(self.config, 'MAX_CLUSTERS', 8),
            'random_state': self.config.RANDOM_STATE,
            'features': self.config.CLUSTERING_FEATURES,
            'feature_weights': getattr(self.config, 'FEATURE_WEIGHTS', {
                'harga': 0.4, 'rating': 0.25, 'jarak': 0.25, 'tipe_encoded': 0.1
            })
        }
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return {
            'prefix': self.config.API_PREFIX,
            'max_results': self.config.MAX_RESULTS_PER_REQUEST,
            'default_limit': self.config.DEFAULT_RESULTS_LIMIT
        }
    
    def get_pagination_config(self) -> Dict[str, Any]:
        """Get pagination configuration"""
        return {
            'per_page': self.config.RESTAURANTS_PER_PAGE,
            'max_per_page': self.config.MAX_RESTAURANTS_PER_PAGE
        }
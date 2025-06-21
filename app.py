from flask import Flask
from config import get_config, AppSettings
from models import RestaurantClusteringModel, RestaurantDataLoader, RestaurantRecommendationEngine
from routes.api_routes import api_bp, init_api_dependencies
from routes.web_routes import web_bp, init_web_dependencies
from utils.logger import get_main_logger
import os

# Initialize logger
logger = get_main_logger()

def create_app(config_name=None):
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Load configuration
    config_obj = get_config(config_name)
    app.config.from_object(config_obj)
    
    # Initialize app settings
    settings = AppSettings(config_obj)
    settings.ensure_directories()
    
    # Validate data file
    if not settings.validate_data_file():
        logger.error(f"Data file not found: {settings.get_data_file_path()}")
        raise FileNotFoundError(f"Data file not found: {settings.get_data_file_path()}")
    
    # Initialize components
    data_loader = RestaurantDataLoader(settings.get_data_file_path())
    clustering_model = RestaurantClusteringModel()
    recommendation_engine = RestaurantRecommendationEngine()
    
    # Load and process data
    try:
        logger.info("Loading and preprocessing data...")
        raw_data = data_loader.load_data()
        processed_data = data_loader.clean_data(raw_data)
        
        logger.info("Performing clustering...")
        clustering_model.fit(processed_data)
        
        # Add cluster labels to data
        clustered_data = processed_data.copy()
        clustered_data['cluster'] = clustering_model.predict(processed_data)
        
        logger.info("Initializing recommendation engine...")
        recommendation_engine.load_data(clustered_data)
        
        logger.info("Application components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing application components: {str(e)}")
        raise
    
    # Initialize route dependencies
    init_api_dependencies(clustering_model, data_loader, recommendation_engine)
    init_web_dependencies(data_loader, clustering_model, recommendation_engine)
    
    # Store processed data for routes
    app.config['PROCESSED_DATA'] = clustered_data
    app.config['DATA_LOADER'] = data_loader
    app.config['CLUSTERING_MODEL'] = clustering_model
    app.config['RECOMMENDATION_ENGINE'] = recommendation_engine
    
    # Register blueprints
    app.register_blueprint(api_bp)
    app.register_blueprint(web_bp)
    
    # Initialize configuration
    config_obj.init_app(app)
    
    return app

# Create application instance
app = create_app()

# Global error handlers
@app.errorhandler(404)
def not_found_error(error):
    return {'error': 'Resource not found'}, 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return {'error': 'Internal server error'}, 500

@app.errorhandler(Exception)
def handle_exception(error):
    logger.error(f"Unhandled exception: {str(error)}")
    return {'error': 'An unexpected error occurred'}, 500

if __name__ == '__main__':
    try:
        logger.info("Starting GachaFood application...")
        
        # Get configuration
        config_obj = get_config()
        
        # Run the application
        if hasattr(config_obj, 'HOST') and hasattr(config_obj, 'PORT'):
            app.run(host=config_obj.HOST, port=config_obj.PORT, debug=config_obj.DEBUG)
        else:
            app.run(host='127.0.0.1', port=8080, debug=True)
            
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise
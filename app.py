from flask import Flask
from config import AppSettings, get_config
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
    
    # Initialize models with improved configuration
    settings = AppSettings(config_obj)
    clustering_config = settings.get_clustering_config()
    
    data_loader = RestaurantDataLoader(settings.get_data_file_path())
    clustering_model = RestaurantClusteringModel(
        n_clusters=clustering_config['n_clusters'],
        auto_optimize=clustering_config['auto_optimize'],
        random_state=clustering_config['random_state']
    )
    # Set feature weights from config
    clustering_model.feature_weights = clustering_config['feature_weights']
    
    recommendation_engine = RestaurantRecommendationEngine()
    
    # Load and process data
    try:
        logger.info("Loading and preprocessing data...")
        raw_data = data_loader.load_data()
        processed_data = data_loader.clean_data(raw_data)
        
        logger.info("Performing clustering...")
        # Fit clustering model
        clustering_model.fit(processed_data)
        
        # Get predictions for the same processed data used in training
        # We need to use the original processed_data and get predictions for it
        # But predict() will preprocess again, so we need to handle this differently
        
        # Get cluster predictions directly from the fitted model
        # First, preprocess the data the same way as in predict method
        prediction_data = clustering_model.preprocess_data(processed_data)
        X = prediction_data[clustering_model.feature_columns]
        
        # Apply feature weights
        X_weighted = X.copy()
        for col in clustering_model.feature_columns:
            if col in clustering_model.feature_weights:
                X_weighted[col] = X_weighted[col] * clustering_model.feature_weights[col]
        
        # Scale and predict
        X_scaled = clustering_model.scaler.transform(X_weighted)
        cluster_predictions = clustering_model.kmeans.predict(X_scaled)
        
        # Use the prediction_data (which has the same length as predictions)
        clustered_data = prediction_data.copy()
        clustered_data['cluster'] = cluster_predictions
        
        # Add cluster names based on price categories
        cluster_means = []
        for cluster_id in range(clustering_model.n_clusters):
            cluster_data = clustered_data[clustered_data['cluster'] == cluster_id]
            if len(cluster_data) > 0:
                cluster_means.append((cluster_id, cluster_data['harga'].mean()))
        
        # Sort by price (ekonomis = lowest price, premium = highest price)
        cluster_means.sort(key=lambda x: x[1])
        
        # Map cluster_id to labels based on price order (dynamic labeling)
        cluster_mapping = {}
        n_clusters = clustering_model.n_clusters
        if n_clusters == 2:
            labels = ['Ekonomis', 'Premium']
        elif n_clusters == 3:
            labels = ['Ekonomis', 'Menengah', 'Premium']  # Gabung sedang dan menengah
        elif n_clusters == 4:
            labels = ['Ekonomis', 'Menengah', 'Tinggi', 'Premium']
        elif n_clusters == 5:
            labels = ['Ekonomis', 'Menengah', 'Tinggi', 'Mewah', 'Premium']
        else:
            labels = [f'Kategori {i+1}' for i in range(n_clusters)]
        
        for i, (cluster_id, _) in enumerate(cluster_means):
            cluster_mapping[cluster_id] = labels[i] if i < len(labels) else f'Cluster {i}'
        
        # Add cluster_name column
        clustered_data['cluster_name'] = clustered_data['cluster'].map(cluster_mapping)
        
        logger.info("Initializing recommendation engine...")
        recommendation_engine.load_data(clustered_data)
        
        # Set clustered data to data loader for API access
        data_loader.set_clustered_data(clustered_data)
        
        logger.info("Application components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing application components: {str(e)}")
        raise
    
    # Initialize route dependencies
    init_api_dependencies(clustering_model, data_loader, recommendation_engine)
    init_web_dependencies(data_loader, clustering_model, recommendation_engine)
    
    # Store components in app config for access by blueprints
    app.config['CLUSTERED_DATA'] = clustered_data
    app.config['PROCESSED_DATA'] = processed_data
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
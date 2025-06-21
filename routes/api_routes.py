from flask import Blueprint, request, jsonify
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Global variables untuk dependency injection
clustering_model = None
data_loader = None
recommendation_engine = None

def init_api_dependencies(clustering, loader, engine):
    """Initialize dependencies untuk API routes"""
    global clustering_model, data_loader, recommendation_engine
    clustering_model = clustering
    data_loader = loader
    recommendation_engine = engine

@api_bp.route('/stats', methods=['GET'])
def get_stats():
    """Endpoint untuk mendapatkan statistik dataset"""
    try:
        if data_loader is None:
            return jsonify({'error': 'Data loader not initialized'}), 500
        
        stats = data_loader.get_summary_stats()
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"Error in get_stats: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/clusters', methods=['GET'])
def get_clusters():
    """Endpoint untuk mendapatkan informasi cluster"""
    try:
        if clustering_model is None:
            return jsonify({'error': 'Clustering model not initialized'}), 500
        
        cluster_info = clustering_model.get_cluster_info()
        return jsonify(cluster_info)
    
    except Exception as e:
        logger.error(f"Error in get_clusters: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/recommendations', methods=['GET'])
def get_recommendations():
    """Endpoint untuk mendapatkan rekomendasi berdasarkan filter"""
    try:
        if recommendation_engine is None:
            return jsonify({'error': 'Recommendation engine not initialized'}), 500
        
        # Parse query parameters
        max_harga = request.args.get('max_harga', type=int)
        max_jarak = request.args.get('max_jarak', type=float)
        min_rating = request.args.get('min_rating', type=float)
        tipe_tempat = request.args.get('tipe_tempat')
        cluster_id = request.args.get('cluster_id', type=int)
        limit = request.args.get('limit', default=10, type=int)
        sort_by = request.args.get('sort_by', default='rating')
        recommendation_type = request.args.get('type', default='filtered')
        
        # Validate limit
        if limit > 50:
            limit = 50
        
        # Get recommendations based on type
        if recommendation_type == 'diverse':
            filters = {
                'max_harga': max_harga,
                'max_jarak': max_jarak,
                'min_rating': min_rating
            }
            recommendations = recommendation_engine.get_diverse_recommendations(
                filters=filters, limit=limit
            )
        elif recommendation_type == 'cluster' and cluster_id is not None:
            recommendations = recommendation_engine.get_cluster_recommendations(
                cluster_id=cluster_id, limit=limit, sort_by=sort_by
            )
        else:
            # Default filtered recommendations
            recommendations = recommendation_engine.get_recommendations_by_filters(
                max_harga=max_harga,
                max_jarak=max_jarak,
                min_rating=min_rating,
                tipe_tempat=tipe_tempat,
                cluster_id=cluster_id,
                limit=limit,
                sort_by=sort_by
            )
        
        return jsonify({
            'recommendations': recommendations,
            'count': len(recommendations),
            'filters_applied': {
                'max_harga': max_harga,
                'max_jarak': max_jarak,
                'min_rating': min_rating,
                'tipe_tempat': tipe_tempat,
                'cluster_id': cluster_id,
                'sort_by': sort_by,
                'type': recommendation_type
            }
        })
    
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/recommendations/personalized', methods=['POST'])
def get_personalized_recommendations():
    """Endpoint untuk mendapatkan rekomendasi yang dipersonalisasi"""
    try:
        if recommendation_engine is None:
            return jsonify({'error': 'Recommendation engine not initialized'}), 500
        
        # Parse JSON body
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        user_preferences = data.get('preferences', {})
        limit = data.get('limit', 10)
        
        # Validate limit
        if limit > 50:
            limit = 50
        
        recommendations = recommendation_engine.get_personalized_recommendations(
            user_preferences=user_preferences,
            limit=limit
        )
        
        return jsonify({
            'recommendations': recommendations,
            'count': len(recommendations),
            'preferences_used': user_preferences
        })
    
    except Exception as e:
        logger.error(f"Error in get_personalized_recommendations: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/restaurant/<int:restaurant_id>', methods=['GET'])
def get_restaurant_details(restaurant_id: int):
    """Endpoint untuk mendapatkan detail restoran dan rekomendasi serupa"""
    try:
        if data_loader is None or recommendation_engine is None:
            return jsonify({'error': 'Services not initialized'}), 500
        
        # Get restaurant details
        restaurant_data = data_loader.get_filtered_data()
        
        if restaurant_id >= len(restaurant_data) or restaurant_id < 0:
            return jsonify({'error': 'Restaurant not found'}), 404
        
        restaurant = restaurant_data.iloc[restaurant_id]
        
        # Get similar restaurants
        n_similar = request.args.get('similar_count', default=5, type=int)
        if n_similar > 10:
            n_similar = 10
        
        similar_restaurants = recommendation_engine.get_similar_restaurants(
            restaurant_id=restaurant_id,
            n_recommendations=n_similar
        )
        
        restaurant_details = {
            'id': restaurant_id,
            'nama_tempat': restaurant['nama_tempat'],
            'tipe_tempat': restaurant['tipe_tempat'],
            'harga': int(restaurant['harga']),
            'rating': float(restaurant['rating']),
            'jarak': float(restaurant['jarak']),
            'lokasi': restaurant['lokasi'],
            'cluster': int(restaurant['cluster']) if 'cluster' in restaurant else None
        }
        
        return jsonify({
            'restaurant': restaurant_details,
            'similar_restaurants': similar_restaurants,
            'similar_count': len(similar_restaurants)
        })
    
    except Exception as e:
        logger.error(f"Error in get_restaurant_details: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/search', methods=['GET'])
def search_restaurants():
    """Endpoint untuk pencarian restoran berdasarkan nama"""
    try:
        if data_loader is None:
            return jsonify({'error': 'Data loader not initialized'}), 500
        
        query = request.args.get('q', '').strip()
        limit = request.args.get('limit', default=10, type=int)
        
        if not query:
            return jsonify({'error': 'Search query is required'}), 400
        
        if limit > 50:
            limit = 50
        
        # Get filtered data
        data = data_loader.get_filtered_data()
        
        # Search by name (case insensitive)
        search_results = data[
            data['nama_tempat'].str.contains(query, case=False, na=False)
        ]
        
        # Sort by rating
        search_results = search_results.sort_values(['rating', 'jarak'], ascending=[False, True])
        
        # Limit results
        search_results = search_results.head(limit)
        
        # Convert to list
        restaurants = []
        for idx, row in search_results.iterrows():
            restaurants.append({
                'id': idx,
                'nama_tempat': row['nama_tempat'],
                'tipe_tempat': row['tipe_tempat'],
                'harga': int(row['harga']),
                'rating': float(row['rating']),
                'jarak': float(row['jarak']),
                'lokasi': row['lokasi'],
                'cluster': int(row['cluster']) if 'cluster' in row else None
            })
        
        return jsonify({
            'restaurants': restaurants,
            'count': len(restaurants),
            'query': query
        })
    
    except Exception as e:
        logger.error(f"Error in search_restaurants: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/clusters/<int:cluster_id>/stats', methods=['GET'])
def get_cluster_stats(cluster_id: int):
    """Endpoint untuk mendapatkan statistik cluster tertentu"""
    try:
        if data_loader is None:
            return jsonify({'error': 'Data loader not initialized'}), 500
        
        data = data_loader.get_filtered_data()
        cluster_data = data[data['cluster'] == cluster_id]
        
        if len(cluster_data) == 0:
            return jsonify({'error': 'Cluster not found'}), 404
        
        stats = {
            'cluster_id': cluster_id,
            'total_restaurants': len(cluster_data),
            'avg_price': float(cluster_data['harga'].mean()),
            'avg_rating': float(cluster_data['rating'].mean()),
            'avg_distance': float(cluster_data['jarak'].mean()),
            'price_range': {
                'min': int(cluster_data['harga'].min()),
                'max': int(cluster_data['harga'].max())
            },
            'rating_range': {
                'min': float(cluster_data['rating'].min()),
                'max': float(cluster_data['rating'].max())
            },
            'distance_range': {
                'min': float(cluster_data['jarak'].min()),
                'max': float(cluster_data['jarak'].max())
            },
            'restaurant_types': cluster_data['tipe_tempat'].value_counts().to_dict()
        }
        
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"Error in get_cluster_stats: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@api_bp.route('/data/quality-report', methods=['GET'])
def get_data_quality_report():
    """Get comprehensive data quality report"""
    try:
        # Get current processed data
        from flask import current_app
        data_loader = current_app.config.get('DATA_LOADER')
        processed_data = current_app.config.get('PROCESSED_DATA')
        
        if data_loader is None or processed_data is None:
            return jsonify({'error': 'Data not available'}), 500
        
        # Generate quality report
        quality_report = data_loader.validate_data_quality(processed_data)
        
        return jsonify({
            'success': True,
            'data': quality_report
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/data/duplicate-report', methods=['GET'])
def get_duplicate_report():
    """Get detailed duplicate detection report"""
    try:
        # Get current processed data
        from flask import current_app
        data_loader = current_app.config.get('DATA_LOADER')
        processed_data = current_app.config.get('PROCESSED_DATA')
        
        if data_loader is None or processed_data is None:
            return jsonify({'error': 'Data not available'}), 500
        
        # Generate duplicate report
        duplicate_report = data_loader.get_duplicate_report(processed_data)
        
        return jsonify({
            'success': True,
            'data': duplicate_report
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Error handlers
@api_bp.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@api_bp.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@api_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500
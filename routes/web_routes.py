from flask import Blueprint, render_template, request, redirect, url_for, flash
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
web_bp = Blueprint('web', __name__)

# Global variables untuk dependency injection
data_loader = None
clustering_model = None
recommendation_engine = None

def init_web_dependencies(loader, clustering, engine):
    """Initialize dependencies untuk web routes"""
    global data_loader, clustering_model, recommendation_engine
    data_loader = loader
    clustering_model = clustering
    recommendation_engine = engine

@web_bp.route('/')
def index():
    """Halaman utama aplikasi"""
    try:
        # Get basic stats untuk ditampilkan di halaman utama
        stats = None
        clusters = None
        
        if data_loader is not None:
            stats = data_loader.get_summary_stats()
        
        if clustering_model is not None and data_loader is not None:
            data_with_clusters = data_loader.get_filtered_data()
            if 'cluster' in data_with_clusters.columns:
                clusters = clustering_model.get_cluster_info(data_with_clusters)
            else:
                clusters = None
        
        return render_template('index.html', stats=stats, clusters=clusters)
    
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        flash('Terjadi kesalahan saat memuat halaman utama', 'error')
        return render_template('index.html', stats=None, clusters=None)

@web_bp.route('/data-quality')
def data_quality():
    """Halaman laporan kualitas data"""
    return render_template('data_quality.html')

@web_bp.route('/restaurants')
def restaurants():
    """Halaman daftar restoran dengan pagination"""
    try:
        if data_loader is None:
            logger.error("Data loader not initialized in restaurants route")
            flash('Data tidak tersedia', 'error')
            return redirect(url_for('web.index'))
        
        logger.info("Processing restaurants page request")
        
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        # Get filter parameters
        max_harga = request.args.get('max_harga', type=int)
        max_jarak = request.args.get('max_jarak', type=float)
        min_rating = request.args.get('min_rating', type=float)
        tipe_tempat = request.args.get('tipe_tempat')
        cluster_id = request.args.get('cluster_id', type=int)
        
        # Get filtered data
        data = data_loader.get_filtered_data()
        
        if data.empty:
            logger.warning("No data available from data loader")
            flash('Data restoran tidak tersedia saat ini', 'warning')
            return redirect(url_for('web.index'))
        
        logger.info(f"Processing {len(data)} restaurants with filters")
        
        # Apply filters
        if max_harga is not None and 'harga' in data.columns:
            data = data[data['harga'] <= max_harga]
        if max_jarak is not None and 'jarak' in data.columns:
            data = data[data['jarak'] <= max_jarak]
        if min_rating is not None and 'rating' in data.columns:
            data = data[data['rating'] >= min_rating]
        if tipe_tempat and 'tipe_tempat' in data.columns:
            data = data[data['tipe_tempat'].str.contains(tipe_tempat, case=False, na=False)]
        if cluster_id is not None and 'cluster' in data.columns:
            data = data[data['cluster'] == cluster_id]
        
        # Sort by rating and distance if columns exist
        sort_columns = []
        sort_ascending = []
        
        if 'rating' in data.columns:
            sort_columns.append('rating')
            sort_ascending.append(False)
        if 'jarak' in data.columns:
            sort_columns.append('jarak')
            sort_ascending.append(True)
            
        if sort_columns:
            data = data.sort_values(sort_columns, ascending=sort_ascending)
        
        logger.info(f"After filtering: {len(data)} restaurants remaining")
        
        # Calculate pagination
        total = len(data)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        # Convert to list of dictionaries for template
        restaurants_page = data.iloc[start_idx:end_idx]
        restaurants = restaurants_page.to_dict('records')
        
        # Add index for template compatibility
        for i, restaurant in enumerate(restaurants):
            restaurant['index'] = start_idx + i
        
        # Calculate pagination info
        has_prev = page > 1
        has_next = end_idx < total
        prev_num = page - 1 if has_prev else None
        next_num = page + 1 if has_next else None
        
        pagination_info = {
            'page': page,
            'per_page': per_page,
            'total': total,
            'pages': (total + per_page - 1) // per_page,
            'has_prev': has_prev,
            'has_next': has_next,
            'prev_num': prev_num,
            'next_num': next_num
        }
        
        # Get available filter options
        all_data = data_loader.get_filtered_data()
        
        if all_data.empty:
            logger.warning("No data available for filter options")
            filter_options = {
                'tipe_tempat': [],
                'clusters': []
            }
        else:
            filter_options = {
                'tipe_tempat': sorted(all_data['tipe_tempat'].unique()) if 'tipe_tempat' in all_data.columns else [],
                'clusters': sorted(all_data['cluster'].unique()) if 'cluster' in all_data.columns else []
            }
        
        logger.info(f"Rendering restaurants page with {len(restaurants)} restaurants")
        
        return render_template(
            'restaurants.html',
            restaurants=restaurants,
            pagination=pagination_info,
            filter_options=filter_options,
            current_filters={
                'max_harga': max_harga,
                'max_jarak': max_jarak,
                'min_rating': min_rating,
                'tipe_tempat': tipe_tempat,
                'cluster_id': cluster_id
            }
        )
    
    except Exception as e:
        logger.error(f"Error in restaurants route: {str(e)}", exc_info=True)
        flash('Terjadi kesalahan saat memuat daftar restoran', 'error')
        return redirect(url_for('web.index'))

@web_bp.route('/restaurant/<int:restaurant_id>')
def restaurant_detail(restaurant_id):
    """Halaman detail restoran"""
    try:
        if data_loader is None or recommendation_engine is None:
            flash('Layanan tidak tersedia', 'error')
            return redirect(url_for('web.index'))
        
        # Get restaurant data
        data = data_loader.get_filtered_data()
        
        if restaurant_id >= len(data) or restaurant_id < 0:
            flash('Restoran tidak ditemukan', 'error')
            return redirect(url_for('web.restaurants'))
        
        restaurant = data.iloc[restaurant_id]
        
        # Get similar restaurants
        similar_restaurants = recommendation_engine.get_similar_restaurants(
            restaurant_id=restaurant_id,
            n_recommendations=5
        )
        
        # Get cluster information if available
        cluster_info = None
        if clustering_model is not None and data_loader is not None and 'cluster' in restaurant:
            data_with_clusters = data_loader.get_filtered_data()
            if 'cluster' in data_with_clusters.columns:
                cluster_data = clustering_model.get_cluster_info(data_with_clusters)
                cluster_info = next(
                    (c for c in cluster_data if c['cluster_id'] == restaurant['cluster']),
                    None
                )
        
        return render_template(
            'restaurant_detail.html',
            restaurant=restaurant,
            restaurant_id=restaurant_id,
            similar_restaurants=similar_restaurants,
            cluster_info=cluster_info
        )
    
    except Exception as e:
        logger.error(f"Error in restaurant_detail route: {str(e)}")
        flash('Terjadi kesalahan saat memuat detail restoran', 'error')
        return redirect(url_for('web.restaurants'))

@web_bp.route('/clusters')
def clusters():
    """Halaman informasi cluster"""
    try:
        if clustering_model is None or data_loader is None:
            flash('Model clustering atau data loader tidak tersedia', 'error')
            return redirect(url_for('web.index'))
        
        data_with_clusters = data_loader.get_filtered_data()
        if 'cluster' not in data_with_clusters.columns:
            flash('Data cluster tidak tersedia', 'error')
            return redirect(url_for('web.index'))
            
        cluster_info = clustering_model.get_cluster_info(data_with_clusters)
        
        return render_template('clusters.html', cluster_info=cluster_info)
    
    except Exception as e:
        logger.error(f"Error in clusters route: {str(e)}")
        flash('Terjadi kesalahan saat memuat informasi cluster', 'error')
        return redirect(url_for('web.index'))

@web_bp.route('/cluster/<int:cluster_id>')
def cluster_detail(cluster_id):
    """Halaman detail cluster"""
    try:
        if data_loader is None or recommendation_engine is None:
            flash('Layanan tidak tersedia', 'error')
            return redirect(url_for('web.index'))
        
        # Get cluster recommendations
        recommendations = recommendation_engine.get_cluster_recommendations(
            cluster_id=cluster_id,
            limit=20,
            sort_by='rating'
        )
        
        if not recommendations:
            flash('Cluster tidak ditemukan', 'error')
            return redirect(url_for('web.clusters'))
        
        # Get cluster stats
        data = data_loader.get_filtered_data()
        cluster_data = data[data['cluster'] == cluster_id]
        
        cluster_stats = {
            'cluster_id': cluster_id,
            'total_restaurants': len(cluster_data),
            'avg_price': cluster_data['harga'].mean(),
            'avg_rating': cluster_data['rating'].mean(),
            'avg_distance': cluster_data['jarak'].mean(),
            'price_range': {
                'min': cluster_data['harga'].min(),
                'max': cluster_data['harga'].max()
            },
            'rating_range': {
                'min': cluster_data['rating'].min(),
                'max': cluster_data['rating'].max()
            },
            'restaurant_types': cluster_data['tipe_tempat'].value_counts().to_dict()
        }
        
        return render_template(
            'cluster_detail.html',
            cluster_id=cluster_id,
            recommendations=recommendations,
            cluster_stats=cluster_stats
        )
    
    except Exception as e:
        logger.error(f"Error in cluster_detail route: {str(e)}")
        flash('Terjadi kesalahan saat memuat detail cluster', 'error')
        return redirect(url_for('web.clusters'))

@web_bp.route('/search')
def search():
    """Halaman pencarian restoran"""
    try:
        query = request.args.get('q', '').strip()
        
        if not query:
            return render_template('search.html', restaurants=[], query='')
        
        if data_loader is None:
            flash('Data tidak tersedia', 'error')
            return render_template('search.html', restaurants=[], query=query)
        
        # Search restaurants
        data = data_loader.get_filtered_data()
        search_results = data[
            data['nama_tempat'].str.contains(query, case=False, na=False)
        ]
        
        # Sort by rating
        search_results = search_results.sort_values(['rating', 'jarak'], ascending=[False, True])
        
        # Limit to 50 results
        search_results = search_results.head(50)
        
        return render_template(
            'search.html',
            restaurants=search_results,
            query=query,
            count=len(search_results)
        )
    
    except Exception as e:
        logger.error(f"Error in search route: {str(e)}")
        flash('Terjadi kesalahan saat melakukan pencarian', 'error')
        return render_template('search.html', restaurants=[], query=query if 'query' in locals() else '')

@web_bp.route('/recommendations')
def recommendations():
    """Halaman rekomendasi dengan berbagai opsi"""
    try:
        if recommendation_engine is None:
            flash('Engine rekomendasi tidak tersedia', 'error')
            return redirect(url_for('web.index'))
        
        recommendation_type = request.args.get('type', 'filtered')
        
        if recommendation_type == 'diverse':
            # Get diverse recommendations
            recommendations = recommendation_engine.get_diverse_recommendations(limit=15)
            title = 'Rekomendasi Beragam'
        else:
            # Get filtered recommendations with default good criteria
            recommendations = recommendation_engine.get_recommendations_by_filters(
                min_rating=4.0,
                limit=15,
                sort_by='hybrid'
            )
            title = 'Rekomendasi Terbaik'
        
        return render_template(
            'recommendations.html',
            recommendations=recommendations,
            title=title,
            recommendation_type=recommendation_type
        )
    
    except Exception as e:
        logger.error(f"Error in recommendations route: {str(e)}")
        flash('Terjadi kesalahan saat memuat rekomendasi', 'error')
        return redirect(url_for('web.index'))

@web_bp.route('/about')
def about():
    """Halaman tentang aplikasi"""
    return render_template('about.html')

# Error handlers
@web_bp.errorhandler(404)
def page_not_found(error):
    return render_template('error.html', 
                         error_code=404, 
                         error_message='Halaman tidak ditemukan'), 404

@web_bp.errorhandler(500)
def internal_server_error(error):
    return render_template('error.html', 
                         error_code=500, 
                         error_message='Terjadi kesalahan internal server'), 500
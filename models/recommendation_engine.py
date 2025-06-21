import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class RestaurantRecommendationEngine:
    """Engine untuk memberikan rekomendasi tempat makan"""
    
    def __init__(self):
        self.data = None
        self.scaler = None
        self.similarity_matrix = None
        
    def load_data(self, df: pd.DataFrame) -> None:
        """Load data yang sudah di-cluster"""
        self.data = df.copy()
        self._prepare_similarity_matrix()
    
    def _prepare_similarity_matrix(self) -> None:
        """Prepare similarity matrix untuk content-based filtering"""
        if self.data is None:
            raise ValueError("Data belum di-load")
        
        # Features untuk similarity calculation
        features = ['harga', 'rating', 'jarak', 'tipe_encoded']
        
        if all(col in self.data.columns for col in features):
            # Normalize features
            self.scaler = StandardScaler()
            feature_matrix = self.scaler.fit_transform(self.data[features])
            
            # Calculate cosine similarity
            self.similarity_matrix = cosine_similarity(feature_matrix)
    
    def get_recommendations_by_filters(
        self, 
        max_harga: Optional[int] = None,
        min_harga: Optional[int] = None,
        max_jarak: Optional[float] = None,
        min_rating: Optional[float] = None,
        tipe_tempat: Optional[str] = None,
        cluster_id: Optional[int] = None,
        cluster_name: Optional[str] = None,
        cluster_names: Optional[str] = None,
        limit: int = 10,
        sort_by: str = 'rating'
    ) -> List[Dict[str, Any]]:
        """Mendapatkan rekomendasi berdasarkan filter"""
        if self.data is None:
            raise ValueError("Data belum di-load")
        
        filtered_df = self.data.copy()
        
        # Apply filters
        if max_harga is not None:
            filtered_df = filtered_df[filtered_df['harga'] <= max_harga]
        
        if min_harga is not None:
            filtered_df = filtered_df[filtered_df['harga'] >= min_harga]
        
        if max_jarak is not None:
            filtered_df = filtered_df[filtered_df['jarak'] <= max_jarak]
        
        if min_rating is not None:
            filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
        
        if tipe_tempat:
            filtered_df = filtered_df[
                filtered_df['tipe_tempat'].str.contains(tipe_tempat, case=False, na=False)
            ]
        
        if cluster_id is not None:
            filtered_df = filtered_df[filtered_df['cluster'] == cluster_id]
        
        # Handle cluster name filtering
        if cluster_name:
            if 'cluster_name' in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df['cluster_name'].str.lower() == cluster_name.lower()
                ]
        
        # Handle multiple cluster names filtering
        if cluster_names:
            cluster_list = [name.strip().lower() for name in cluster_names.split(',')]
            if 'cluster_name' in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df['cluster_name'].str.lower().isin(cluster_list)
                ]
        
        # Sort results
        if sort_by == 'rating':
            filtered_df = filtered_df.sort_values(['rating', 'jarak'], ascending=[False, True])
        elif sort_by == 'distance':
            filtered_df = filtered_df.sort_values(['jarak', 'rating'], ascending=[True, False])
        elif sort_by == 'price':
            filtered_df = filtered_df.sort_values(['harga', 'rating'], ascending=[True, False])
        elif sort_by == 'hybrid':
            # Hybrid scoring: kombinasi rating, jarak, dan harga
            filtered_df = self._calculate_hybrid_score(filtered_df)
            filtered_df = filtered_df.sort_values('hybrid_score', ascending=False)
        
        # Limit results
        filtered_df = filtered_df.head(limit)
        
        # Convert to list of dictionaries
        recommendations = []
        for _, row in filtered_df.iterrows():
            recommendations.append({
                'nama_tempat': row['nama_tempat'],
                'tipe_tempat': row['tipe_tempat'],
                'harga': int(row['harga']),
                'rating': float(row['rating']),
                'jarak': float(row['jarak']),
                'lokasi': row['lokasi'],
                'cluster': int(row['cluster']) if 'cluster' in row else None,
                'hybrid_score': float(row['hybrid_score']) if 'hybrid_score' in row else None
            })
        
        return recommendations
    
    def _calculate_hybrid_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate hybrid score berdasarkan multiple criteria"""
        df_scored = df.copy()
        
        # Normalize scores to 0-1 range
        # Rating score (higher is better)
        rating_score = (df_scored['rating'] - df_scored['rating'].min()) / \
                      (df_scored['rating'].max() - df_scored['rating'].min())
        
        # Distance score (lower is better, so invert)
        distance_score = 1 - ((df_scored['jarak'] - df_scored['jarak'].min()) / \
                             (df_scored['jarak'].max() - df_scored['jarak'].min()))
        
        # Price score (lower is better, so invert)
        price_score = 1 - ((df_scored['harga'] - df_scored['harga'].min()) / \
                          (df_scored['harga'].max() - df_scored['harga'].min()))
        
        # Weighted combination (dapat disesuaikan)
        weights = {'rating': 0.4, 'distance': 0.35, 'price': 0.25}
        
        df_scored['hybrid_score'] = (
            weights['rating'] * rating_score +
            weights['distance'] * distance_score +
            weights['price'] * price_score
        )
        
        return df_scored
    
    def get_similar_restaurants(
        self, 
        restaurant_id: int, 
        n_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """Mendapatkan restoran yang mirip berdasarkan content-based filtering"""
        if self.data is None or self.similarity_matrix is None:
            raise ValueError("Data atau similarity matrix belum di-prepare")
        
        if restaurant_id >= len(self.data) or restaurant_id < 0:
            raise ValueError("Restaurant ID tidak valid")
        
        # Get similarity scores for the restaurant
        sim_scores = list(enumerate(self.similarity_matrix[restaurant_id]))
        
        # Sort by similarity score (excluding the restaurant itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]
        
        # Get top N similar restaurants
        similar_indices = [i[0] for i in sim_scores[:n_recommendations]]
        
        # Get restaurant details
        similar_restaurants = []
        for idx in similar_indices:
            row = self.data.iloc[idx]
            similar_restaurants.append({
                'nama_tempat': row['nama_tempat'],
                'tipe_tempat': row['tipe_tempat'],
                'harga': int(row['harga']),
                'rating': float(row['rating']),
                'jarak': float(row['jarak']),
                'lokasi': row['lokasi'],
                'cluster': int(row['cluster']) if 'cluster' in row else None,
                'similarity_score': float(sim_scores[similar_indices.index(idx)][1])
            })
        
        return similar_restaurants
    
    def get_cluster_recommendations(
        self, 
        cluster_id: int, 
        limit: int = 10,
        sort_by: str = 'rating'
    ) -> List[Dict[str, Any]]:
        """Mendapatkan rekomendasi dari cluster tertentu"""
        if self.data is None:
            raise ValueError("Data belum di-load")
        
        cluster_data = self.data[self.data['cluster'] == cluster_id]
        
        if len(cluster_data) == 0:
            return []
        
        # Sort based on criteria
        if sort_by == 'rating':
            cluster_data = cluster_data.sort_values(['rating', 'jarak'], ascending=[False, True])
        elif sort_by == 'distance':
            cluster_data = cluster_data.sort_values(['jarak', 'rating'], ascending=[True, False])
        elif sort_by == 'price':
            cluster_data = cluster_data.sort_values(['harga', 'rating'], ascending=[True, False])
        
        # Limit results
        cluster_data = cluster_data.head(limit)
        
        # Convert to list
        recommendations = []
        for _, row in cluster_data.iterrows():
            recommendations.append({
                'nama_tempat': row['nama_tempat'],
                'tipe_tempat': row['tipe_tempat'],
                'harga': int(row['harga']),
                'rating': float(row['rating']),
                'jarak': float(row['jarak']),
                'lokasi': row['lokasi'],
                'cluster': int(row['cluster'])
            })
        
        return recommendations
    
    def get_personalized_recommendations(
        self, 
        user_preferences: Dict[str, Any], 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Mendapatkan rekomendasi yang dipersonalisasi berdasarkan preferensi user"""
        if self.data is None:
            raise ValueError("Data belum di-load")
        
        # Extract user preferences
        preferred_price_range = user_preferences.get('price_range', [0, float('inf')])
        preferred_distance_max = user_preferences.get('max_distance', float('inf'))
        preferred_rating_min = user_preferences.get('min_rating', 0)
        preferred_types = user_preferences.get('restaurant_types', [])
        price_weight = user_preferences.get('price_weight', 0.3)
        distance_weight = user_preferences.get('distance_weight', 0.3)
        rating_weight = user_preferences.get('rating_weight', 0.4)
        
        # Filter based on hard constraints
        filtered_df = self.data[
            (self.data['harga'] >= preferred_price_range[0]) &
            (self.data['harga'] <= preferred_price_range[1]) &
            (self.data['jarak'] <= preferred_distance_max) &
            (self.data['rating'] >= preferred_rating_min)
        ]
        
        if preferred_types:
            type_filter = filtered_df['tipe_tempat'].isin(preferred_types)
            filtered_df = filtered_df[type_filter]
        
        if len(filtered_df) == 0:
            return []
        
        # Calculate personalized scores
        filtered_df = filtered_df.copy()
        
        # Normalize scores
        price_score = 1 - ((filtered_df['harga'] - filtered_df['harga'].min()) / 
                          (filtered_df['harga'].max() - filtered_df['harga'].min() + 1e-8))
        distance_score = 1 - ((filtered_df['jarak'] - filtered_df['jarak'].min()) / 
                             (filtered_df['jarak'].max() - filtered_df['jarak'].min() + 1e-8))
        rating_score = (filtered_df['rating'] - filtered_df['rating'].min()) / \
                      (filtered_df['rating'].max() - filtered_df['rating'].min() + 1e-8)
        
        # Calculate weighted personalized score
        filtered_df['personalized_score'] = (
            price_weight * price_score +
            distance_weight * distance_score +
            rating_weight * rating_score
        )
        
        # Sort by personalized score
        filtered_df = filtered_df.sort_values('personalized_score', ascending=False)
        
        # Limit results
        filtered_df = filtered_df.head(limit)
        
        # Convert to list
        recommendations = []
        for _, row in filtered_df.iterrows():
            recommendations.append({
                'nama_tempat': row['nama_tempat'],
                'tipe_tempat': row['tipe_tempat'],
                'harga': int(row['harga']),
                'rating': float(row['rating']),
                'jarak': float(row['jarak']),
                'lokasi': row['lokasi'],
                'cluster': int(row['cluster']) if 'cluster' in row else None,
                'personalized_score': float(row['personalized_score'])
            })
        
        return recommendations
    
    def get_diverse_recommendations(
        self, 
        filters: Dict[str, Any] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Mendapatkan rekomendasi yang beragam dari berbagai cluster"""
        if self.data is None:
            raise ValueError("Data belum di-load")
        
        # Apply basic filters if provided
        filtered_df = self.data.copy()
        if filters:
            if 'max_harga' in filters and filters['max_harga']:
                filtered_df = filtered_df[filtered_df['harga'] <= filters['max_harga']]
            if 'max_jarak' in filters and filters['max_jarak']:
                filtered_df = filtered_df[filtered_df['jarak'] <= filters['max_jarak']]
            if 'min_rating' in filters and filters['min_rating']:
                filtered_df = filtered_df[filtered_df['rating'] >= filters['min_rating']]
        
        if len(filtered_df) == 0:
            return []
        
        # Get recommendations from each cluster
        diverse_recommendations = []
        clusters = filtered_df['cluster'].unique()
        
        # Calculate how many recommendations per cluster
        recs_per_cluster = max(1, limit // len(clusters))
        remaining_slots = limit - (recs_per_cluster * len(clusters))
        
        for cluster_id in clusters:
            cluster_data = filtered_df[filtered_df['cluster'] == cluster_id]
            cluster_data = cluster_data.sort_values(['rating', 'jarak'], ascending=[False, True])
            
            # Take top restaurants from this cluster
            cluster_recs = cluster_data.head(recs_per_cluster)
            
            for _, row in cluster_recs.iterrows():
                diverse_recommendations.append({
                    'nama_tempat': row['nama_tempat'],
                    'tipe_tempat': row['tipe_tempat'],
                    'harga': int(row['harga']),
                    'rating': float(row['rating']),
                    'jarak': float(row['jarak']),
                    'lokasi': row['lokasi'],
                    'cluster': int(row['cluster'])
                })
        
        # Fill remaining slots with top-rated restaurants
        if remaining_slots > 0:
            remaining_data = filtered_df[
                ~filtered_df['nama_tempat'].isin([r['nama_tempat'] for r in diverse_recommendations])
            ]
            remaining_data = remaining_data.sort_values(['rating', 'jarak'], ascending=[False, True])
            
            for _, row in remaining_data.head(remaining_slots).iterrows():
                diverse_recommendations.append({
                    'nama_tempat': row['nama_tempat'],
                    'tipe_tempat': row['tipe_tempat'],
                    'harga': int(row['harga']),
                    'rating': float(row['rating']),
                    'jarak': float(row['jarak']),
                    'lokasi': row['lokasi'],
                    'cluster': int(row['cluster'])
                })
        
        return diverse_recommendations[:limit]
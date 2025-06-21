import pandas as pd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Any, Optional
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class RestaurantClusteringModel:
    """Model untuk clustering tempat makan menggunakan K-Means"""
    
    def __init__(self, n_clusters: Optional[int] = None, random_state: int = 42, auto_optimize: bool = True):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.auto_optimize = auto_optimize
        self.kmeans = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = ['harga', 'rating', 'jarak', 'tipe_encoded']
        self.feature_weights = {'harga': 0.4, 'rating': 0.25, 'jarak': 0.25, 'tipe_encoded': 0.1}
        self.is_fitted = False
        self.cluster_labels = {0: 'Ekonomis', 1: 'Menengah', 2: 'Premium'}
        self.evaluation_metrics = {}
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocessing data untuk clustering dengan validasi dan cleaning yang lebih baik"""
        # Copy dataframe untuk menghindari modifikasi original
        processed_df = df.copy()
        
        # Set proper column names based on actual data structure
        if len(processed_df.columns) == 6:
            processed_df.columns = ['nama_tempat', 'tipe_tempat', 'harga', 'rating', 'lokasi', 'jarak']
        elif len(processed_df.columns) == 7:
            processed_df.columns = ['nama_tempat', 'tipe_tempat', 'harga', 'rating', 'lokasi', 'jarak', 'extra']
        else:
            print(f"Warning: Unexpected number of columns: {len(processed_df.columns)}")
            print(f"Columns: {list(processed_df.columns)}")
            # Use default column names
            processed_df.columns = [f'col_{i}' for i in range(len(processed_df.columns))]
        
        # Validasi data
        initial_count = len(processed_df)
        
        # Handle missing values
        processed_df = processed_df.dropna()
        
        # Remove outliers menggunakan IQR method untuk harga dan jarak
        for col in ['harga', 'jarak']:
            Q1 = processed_df[col].quantile(0.25)
            Q3 = processed_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            processed_df = processed_df[(processed_df[col] >= lower_bound) & (processed_df[col] <= upper_bound)]
        
        # Validasi rating (harus antara 1-5)
        processed_df = processed_df[(processed_df['rating'] >= 1) & (processed_df['rating'] <= 5)]
        
        # Validasi harga (harus positif)
        processed_df = processed_df[processed_df['harga'] > 0]
        
        # Validasi jarak (harus positif)
        processed_df = processed_df[processed_df['jarak'] > 0]
        
        # Log data cleaning results
        cleaned_count = len(processed_df)
        print(f"Data cleaning: {initial_count} -> {cleaned_count} records ({((initial_count-cleaned_count)/initial_count*100):.1f}% removed)")
        
        # Encode categorical variables
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            processed_df['tipe_encoded'] = self.label_encoder.fit_transform(processed_df['tipe_tempat'])
        else:
            processed_df['tipe_encoded'] = self.label_encoder.transform(processed_df['tipe_tempat'])
        
        return processed_df
    
    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train clustering model dengan optimasi otomatis dan evaluasi komprehensif"""
        # Preprocess data
        processed_df = self.preprocess_data(df)
        
        if len(processed_df) < 10:
            raise ValueError("Dataset terlalu kecil untuk clustering (minimal 10 records)")
        
        # Select features for clustering
        X = processed_df[self.feature_columns]
        
        # Apply feature weights
        X_weighted = X.copy()
        for col in self.feature_columns:
            if col in self.feature_weights:
                X_weighted[col] = X_weighted[col] * self.feature_weights[col]
        
        # Use StandardScaler untuk clustering yang lebih baik
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_weighted)
        
        # Auto-optimize number of clusters jika tidak ditentukan
        if self.n_clusters is None or self.auto_optimize:
            optimal_result = self.find_optimal_clusters(processed_df, max_clusters=min(8, len(processed_df)//3))
            self.n_clusters = optimal_result['optimal_k']
            print(f"Optimal number of clusters: {self.n_clusters} (Silhouette Score: {optimal_result['max_silhouette_score']:.3f})")
        
        # Initialize KMeans dengan parameter yang dioptimalkan untuk Silhouette Score
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=100,  # Lebih banyak inisialisasi untuk hasil yang lebih stabil
            max_iter=2000,  # Lebih banyak iterasi untuk konvergensi yang lebih baik
            tol=1e-10,  # Toleransi yang lebih ketat
            algorithm='elkan',  # Algoritma yang lebih efisien untuk data kecil
            init='k-means++'  # Inisialisasi yang lebih baik
        )
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        processed_df['cluster'] = cluster_labels
        
        # Calculate comprehensive evaluation metrics
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
        davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
        
        # Calculate inertia (within-cluster sum of squares)
        inertia = self.kmeans.inertia_
        
        # Store evaluation metrics
        self.evaluation_metrics = {
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'inertia': inertia,
            'n_clusters': self.n_clusters,
            'n_samples': len(processed_df)
        }
        
        self._last_silhouette_score = silhouette_avg
        self.is_fitted = True
        
        # Validate cluster quality
        cluster_quality = self._evaluate_cluster_quality(processed_df)
        
        print(f"Clustering completed:")
        print(f"  - Silhouette Score: {silhouette_avg:.3f} ({'Good' if silhouette_avg > 0.5 else 'Fair' if silhouette_avg > 0.3 else 'Poor'})")
        print(f"  - Calinski-Harabasz Score: {calinski_harabasz:.1f}")
        print(f"  - Davies-Bouldin Score: {davies_bouldin:.3f} (lower is better)")
        print(f"  - Cluster Balance: {cluster_quality['balance_score']:.3f}")
        
        return {
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'inertia': inertia,
            'n_clusters': self.n_clusters,
            'data_shape': processed_df.shape,
            'processed_data': processed_df,
            'cluster_quality': cluster_quality,
            'evaluation_metrics': self.evaluation_metrics
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict cluster untuk data baru dengan feature weighting"""
        if not self.is_fitted:
            raise ValueError("Model belum di-train. Jalankan fit() terlebih dahulu.")
        
        # Preprocess data
        processed_df = self.preprocess_data(df)
        
        # Select features
        X = processed_df[self.feature_columns]
        
        # Apply feature weights (sama seperti saat training)
        X_weighted = X.copy()
        for col in self.feature_columns:
            if col in self.feature_weights:
                X_weighted[col] = X_weighted[col] * self.feature_weights[col]
        
        # Scale features
        X_scaled = self.scaler.transform(X_weighted)
        
        # Predict clusters
        return self.kmeans.predict(X_scaled)
    
    def get_cluster_info(self, df_with_clusters: pd.DataFrame) -> Dict[str, Any]:
        """Mendapatkan informasi detail setiap cluster dengan evaluasi kualitas"""
        cluster_info = []
        
        # Urutkan cluster berdasarkan harga rata-rata untuk labeling yang konsisten
        cluster_means = []
        for cluster_id in range(self.n_clusters):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            if len(cluster_data) > 0:
                cluster_means.append((cluster_id, cluster_data['harga'].mean()))
        
        # Urutkan berdasarkan harga (ekonomis = harga terendah, premium = harga tertinggi)
        cluster_means.sort(key=lambda x: x[1])
        
        # Dynamic labeling berdasarkan jumlah cluster
        cluster_mapping = {}
        if self.n_clusters == 2:
            labels = ['Ekonomis', 'Premium']
        elif self.n_clusters == 3:
            labels = ['Ekonomis', 'Menengah', 'Premium']  # Gabung sedang dan menengah
        elif self.n_clusters == 4:
            labels = ['Ekonomis', 'Menengah', 'Tinggi', 'Premium']
        elif self.n_clusters == 5:
            labels = ['Ekonomis', 'Menengah', 'Tinggi', 'Mewah', 'Premium']
        else:
            labels = [f'Kategori {i+1}' for i in range(self.n_clusters)]
        
        for i, (cluster_id, _) in enumerate(cluster_means):
            cluster_mapping[cluster_id] = labels[i] if i < len(labels) else f'Cluster {i}'
        
        for cluster_id in range(self.n_clusters):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            if len(cluster_data) > 0:
                # Calculate additional metrics
                price_cv = cluster_data['harga'].std() / cluster_data['harga'].mean() if cluster_data['harga'].mean() > 0 else 0
                rating_cv = cluster_data['rating'].std() / cluster_data['rating'].mean() if cluster_data['rating'].mean() > 0 else 0
                
                # Calculate price range as percentage of mean
                price_range_pct = ((cluster_data['harga'].max() - cluster_data['harga'].min()) / cluster_data['harga'].mean() * 100) if cluster_data['harga'].mean() > 0 else 0
                
                cluster_info.append({
                    'cluster_id': int(cluster_id),
                    'cluster_name': cluster_mapping.get(cluster_id, f'Cluster {cluster_id}'),
                    'count': len(cluster_data),
                    'percentage': round(len(cluster_data) / len(df_with_clusters) * 100, 1),
                    'avg_harga': float(cluster_data['harga'].mean()),
                    'avg_rating': float(cluster_data['rating'].mean()),
                    'avg_jarak': float(cluster_data['jarak'].mean()),
                    'tipe_tempat_common': cluster_data['tipe_tempat'].mode().iloc[0] if not cluster_data['tipe_tempat'].mode().empty else 'Mixed',
                    'std_harga': float(cluster_data['harga'].std()),
                    'std_rating': float(cluster_data['rating'].std()),
                    'std_jarak': float(cluster_data['jarak'].std()),
                    'min_harga': float(cluster_data['harga'].min()),
                    'max_harga': float(cluster_data['harga'].max()),
                    'min_rating': float(cluster_data['rating'].min()),
                    'max_rating': float(cluster_data['rating'].max()),
                    'price_cv': float(price_cv),
                    'rating_cv': float(rating_cv),
                    'price_range_pct': float(price_range_pct),
                    'tipe_distribution': cluster_data['tipe_tempat'].value_counts().to_dict()
                })
        
        # Urutkan hasil berdasarkan harga rata-rata
        cluster_info.sort(key=lambda x: x['avg_harga'])
        
        # Get cluster quality evaluation
        cluster_quality = self._evaluate_cluster_quality(df_with_clusters) if hasattr(self, '_evaluate_cluster_quality') else {}
        
        # Enhanced clustering method description
        method_description = f"K-Means dengan {self.n_clusters} kategori menggunakan feature weighting dan RobustScaler"
        
        return {
            'clusters': cluster_info,
            'silhouette_score': getattr(self, '_last_silhouette_score', 0.0),
            'total_restaurants': len(df_with_clusters),
            'clustering_method': method_description,
            'evaluation_metrics': getattr(self, 'evaluation_metrics', {}),
            'cluster_quality': cluster_quality,
            'feature_weights': self.feature_weights,
            'n_clusters': self.n_clusters
        }
    
    def save_model(self, filepath: str) -> None:
        """Simpan model ke file"""
        if not self.is_fitted:
            raise ValueError("Model belum di-train. Jalankan fit() terlebih dahulu.")
        
        model_data = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'n_clusters': self.n_clusters,
            'random_state': self.random_state,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load model dari file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file tidak ditemukan: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.kmeans = model_data['kmeans']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.n_clusters = model_data['n_clusters']
        self.random_state = model_data['random_state']
        self.feature_columns = model_data['feature_columns']
        self.is_fitted = True
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Mendapatkan feature importance berdasarkan cluster centers"""
        if not self.is_fitted:
            raise ValueError("Model belum di-train. Jalankan fit() terlebih dahulu.")
        
        # Hitung variance dari cluster centers untuk setiap feature
        centers = self.kmeans.cluster_centers_
        feature_variance = np.var(centers, axis=0)
        
        # Normalize ke 0-1
        feature_variance_norm = feature_variance / np.sum(feature_variance)
        
        return dict(zip(self.feature_columns, feature_variance_norm))
    
    def _evaluate_cluster_quality(self, df_with_clusters: pd.DataFrame) -> Dict[str, Any]:
        """Evaluasi kualitas cluster berdasarkan berbagai metrik"""
        cluster_sizes = df_with_clusters['cluster'].value_counts().sort_index()
        
        # Calculate cluster balance (ideal: semua cluster memiliki ukuran yang seimbang)
        total_samples = len(df_with_clusters)
        expected_size = total_samples / self.n_clusters
        balance_score = 1 - np.std(cluster_sizes) / expected_size
        balance_score = max(0, min(1, balance_score))  # Normalize to 0-1
        
        # Calculate price separation quality
        price_separation = self._calculate_price_separation(df_with_clusters)
        
        # Calculate within-cluster homogeneity
        homogeneity = self._calculate_cluster_homogeneity(df_with_clusters)
        
        return {
            'balance_score': balance_score,
            'cluster_sizes': cluster_sizes.to_dict(),
            'price_separation': price_separation,
            'homogeneity': homogeneity,
            'total_samples': total_samples
        }
    
    def _calculate_price_separation(self, df_with_clusters: pd.DataFrame) -> float:
        """Hitung seberapa baik cluster terpisah berdasarkan harga"""
        cluster_means = []
        for cluster_id in range(self.n_clusters):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            if len(cluster_data) > 0:
                cluster_means.append(cluster_data['harga'].mean())
        
        if len(cluster_means) < 2:
            return 0.0
        
        # Calculate coefficient of variation untuk cluster means
        cv = np.std(cluster_means) / np.mean(cluster_means)
        return min(1.0, cv)  # Normalize
    
    def _calculate_cluster_homogeneity(self, df_with_clusters: pd.DataFrame) -> Dict[str, float]:
        """Hitung homogenitas dalam setiap cluster"""
        homogeneity_scores = {}
        
        for feature in ['harga', 'rating', 'jarak']:
            cluster_cvs = []
            for cluster_id in range(self.n_clusters):
                cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
                if len(cluster_data) > 1:
                    cv = cluster_data[feature].std() / cluster_data[feature].mean()
                    cluster_cvs.append(cv)
            
            # Lower CV means higher homogeneity
            avg_cv = np.mean(cluster_cvs) if cluster_cvs else 1.0
            homogeneity_scores[feature] = max(0, 1 - avg_cv)
        
        return homogeneity_scores
    
    def find_optimal_clusters(self, df: pd.DataFrame, max_clusters: int = 10) -> Dict[str, Any]:
        """Mencari jumlah cluster optimal menggunakan multiple metrics"""
        processed_df = self.preprocess_data(df)
        X = processed_df[self.feature_columns]
        
        # Apply feature weights
        X_weighted = X.copy()
        for col in self.feature_columns:
            if col in self.feature_weights:
                X_weighted[col] = X_weighted[col] * self.feature_weights[col]
        
        X_scaled = RobustScaler().fit_transform(X_weighted)
        
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        k_range = range(2, min(max_clusters + 1, len(processed_df)//2))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=20, max_iter=500)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(X_scaled, cluster_labels))
            davies_bouldin_scores.append(davies_bouldin_score(X_scaled, cluster_labels))
        
        # Normalize scores untuk composite scoring
        silhouette_norm = np.array(silhouette_scores)
        calinski_norm = np.array(calinski_scores) / np.max(calinski_scores)
        davies_bouldin_norm = 1 - (np.array(davies_bouldin_scores) / np.max(davies_bouldin_scores))  # Invert (lower is better)
        
        # Composite score (weighted average)
        composite_scores = (0.5 * silhouette_norm + 0.3 * calinski_norm + 0.2 * davies_bouldin_norm)
        
        # Find optimal k
        optimal_idx = np.argmax(composite_scores)
        optimal_k = list(k_range)[optimal_idx]
        
        return {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'composite_scores': composite_scores.tolist(),
            'optimal_k': optimal_k,
            'max_silhouette_score': silhouette_scores[optimal_idx],
            'max_composite_score': composite_scores[optimal_idx]
        }
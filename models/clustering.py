import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from typing import Tuple, Dict, Any
import joblib
import os

class RestaurantClusteringModel:
    """Model untuk clustering tempat makan menggunakan K-Means"""
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = ['harga', 'rating', 'jarak', 'tipe_encoded']
        self.is_fitted = False
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocessing data untuk clustering"""
        # Copy dataframe untuk menghindari modifikasi original
        processed_df = df.copy()
        
        # Clean column names
        processed_df.columns = ['nama_tempat', 'tipe_tempat', 'harga', 'rating', 'lokasi', 'jarak']
        
        # Handle missing values
        processed_df = processed_df.dropna()
        
        # Encode categorical variables
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            processed_df['tipe_encoded'] = self.label_encoder.fit_transform(processed_df['tipe_tempat'])
        else:
            processed_df['tipe_encoded'] = self.label_encoder.transform(processed_df['tipe_tempat'])
        
        return processed_df
    
    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train clustering model"""
        # Preprocess data
        processed_df = self.preprocess_data(df)
        
        # Select features for clustering
        X = processed_df[self.feature_columns]
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform K-means clustering
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_state, 
            n_init=10
        )
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        processed_df['cluster'] = cluster_labels
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        
        self.is_fitted = True
        
        return {
            'silhouette_score': silhouette_avg,
            'n_clusters': self.n_clusters,
            'data_shape': processed_df.shape,
            'processed_data': processed_df
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict cluster untuk data baru"""
        if not self.is_fitted:
            raise ValueError("Model belum di-train. Jalankan fit() terlebih dahulu.")
        
        # Preprocess data
        processed_df = self.preprocess_data(df)
        
        # Select features
        X = processed_df[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict clusters
        return self.kmeans.predict(X_scaled)
    
    def get_cluster_info(self, df_with_clusters: pd.DataFrame) -> Dict[str, Any]:
        """Mendapatkan informasi detail setiap cluster"""
        cluster_info = []
        
        for cluster_id in range(self.n_clusters):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            if len(cluster_data) > 0:
                cluster_info.append({
                    'cluster_id': int(cluster_id),
                    'count': len(cluster_data),
                    'avg_harga': float(cluster_data['harga'].mean()),
                    'avg_rating': float(cluster_data['rating'].mean()),
                    'avg_jarak': float(cluster_data['jarak'].mean()),
                    'tipe_tempat_common': cluster_data['tipe_tempat'].mode().iloc[0] if not cluster_data['tipe_tempat'].mode().empty else 'Mixed',
                    'std_harga': float(cluster_data['harga'].std()),
                    'std_rating': float(cluster_data['rating'].std()),
                    'std_jarak': float(cluster_data['jarak'].std())
                })
        
        return cluster_info
    
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
    
    def find_optimal_clusters(self, df: pd.DataFrame, max_clusters: int = 10) -> Dict[str, Any]:
        """Mencari jumlah cluster optimal menggunakan elbow method dan silhouette score"""
        processed_df = self.preprocess_data(df)
        X = processed_df[self.feature_columns]
        X_scaled = StandardScaler().fit_transform(X)
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
        
        # Temukan optimal k berdasarkan silhouette score tertinggi
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        return {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k': optimal_k,
            'max_silhouette_score': max(silhouette_scores)
        }
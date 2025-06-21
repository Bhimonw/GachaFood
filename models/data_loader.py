import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import os
from pathlib import Path
from utils.duplicate_detector import DuplicateDetector

class RestaurantDataLoader:
    """Class untuk loading dan preprocessing data tempat makan"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load data dari CSV file"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file tidak ditemukan: {self.data_path}")
        
        try:
            self.raw_data = pd.read_csv(self.data_path)
            return self.raw_data.copy()
        except Exception as e:
            raise ValueError(f"Error saat membaca file CSV: {str(e)}")
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validasi struktur dan kualitas data"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check required columns
        required_columns = ['Nama Tempat', 'Tipe Tempat', 'Estimasi Harga (RP.)', 
                          'Rating Tempat Makan', 'Lokasi (maps)', 'Jarak dari kampus (km)']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Kolom yang hilang: {missing_columns}")
        
        if validation_result['is_valid']:
            # Check data types and ranges
            try:
                # Harga harus numerik dan positif
                harga_col = 'Estimasi Harga (RP.)'
                if harga_col in df.columns:
                    harga_numeric = pd.to_numeric(df[harga_col], errors='coerce')
                    invalid_harga = harga_numeric.isna().sum()
                    negative_harga = (harga_numeric < 0).sum()
                    
                    if invalid_harga > 0:
                        validation_result['warnings'].append(f"{invalid_harga} baris dengan harga tidak valid")
                    if negative_harga > 0:
                        validation_result['warnings'].append(f"{negative_harga} baris dengan harga negatif")
                
                # Rating harus antara 1-5
                rating_col = 'Rating Tempat Makan'
                if rating_col in df.columns:
                    rating_numeric = pd.to_numeric(df[rating_col], errors='coerce')
                    invalid_rating = rating_numeric.isna().sum()
                    out_of_range_rating = ((rating_numeric < 1) | (rating_numeric > 5)).sum()
                    
                    if invalid_rating > 0:
                        validation_result['warnings'].append(f"{invalid_rating} baris dengan rating tidak valid")
                    if out_of_range_rating > 0:
                        validation_result['warnings'].append(f"{out_of_range_rating} baris dengan rating di luar range 1-5")
                
                # Jarak harus numerik dan positif
                jarak_col = 'Jarak dari kampus (km)'
                if jarak_col in df.columns:
                    jarak_numeric = pd.to_numeric(df[jarak_col], errors='coerce')
                    invalid_jarak = jarak_numeric.isna().sum()
                    negative_jarak = (jarak_numeric < 0).sum()
                    
                    if invalid_jarak > 0:
                        validation_result['warnings'].append(f"{invalid_jarak} baris dengan jarak tidak valid")
                    if negative_jarak > 0:
                        validation_result['warnings'].append(f"{negative_jarak} baris dengan jarak negatif")
                        
            except Exception as e:
                validation_result['warnings'].append(f"Error saat validasi data: {str(e)}")
        
        # Generate statistics
        validation_result['stats'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        return validation_result
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Membersihkan dan preprocessing data"""
        cleaned_df = df.copy()
        
        # Standardize column names
        column_mapping = {
            'Nama Tempat': 'nama_tempat',
            'Tipe Tempat': 'tipe_tempat',
            'Estimasi Harga (RP.)': 'harga',
            'Rating Tempat Makan': 'rating',
            'Lokasi (maps)': 'lokasi',
            'Jarak dari kampus (km)': 'jarak'
        }
        
        cleaned_df = cleaned_df.rename(columns=column_mapping)
        
        # Convert data types
        try:
            cleaned_df['harga'] = pd.to_numeric(cleaned_df['harga'], errors='coerce')
            cleaned_df['rating'] = pd.to_numeric(cleaned_df['rating'], errors='coerce')
            cleaned_df['jarak'] = pd.to_numeric(cleaned_df['jarak'], errors='coerce')
        except Exception as e:
            raise ValueError(f"Error saat konversi tipe data: {str(e)}")
        
        # Remove rows with invalid data
        initial_rows = len(cleaned_df)
        
        # Remove rows with missing critical values
        cleaned_df = cleaned_df.dropna(subset=['nama_tempat', 'harga', 'rating', 'jarak'])
        
        # Remove rows with invalid ranges
        cleaned_df = cleaned_df[
            (cleaned_df['harga'] > 0) & 
            (cleaned_df['rating'] >= 1) & (cleaned_df['rating'] <= 5) &
            (cleaned_df['jarak'] >= 0)
        ]
        
        # Clean text fields
        cleaned_df['nama_tempat'] = cleaned_df['nama_tempat'].str.strip()
        cleaned_df['tipe_tempat'] = cleaned_df['tipe_tempat'].str.strip()
        
        # Standardize tipe_tempat values
        tipe_mapping = {
            'indoor': 'Indoor',
            'outdoor': 'Outdoor',
            'warung': 'Warung',
            'cafe': 'Cafe',
            'mix': 'Mix'
        }
        
        for old_val, new_val in tipe_mapping.items():
            cleaned_df['tipe_tempat'] = cleaned_df['tipe_tempat'].str.replace(
                old_val, new_val, case=False, regex=False
            )
        
        # Advanced duplicate removal using DuplicateDetector
        duplicate_detector = DuplicateDetector(similarity_threshold=0.85)
        
        # Remove duplicates with advanced similarity matching
        cleaned_df = duplicate_detector.remove_duplicates_advanced(
            cleaned_df, 
            name_col='nama_tempat', 
            location_col='lokasi',
            rating_col='rating'
        )
        
        final_rows = len(cleaned_df)
        removed_rows = initial_rows - final_rows
        
        self.processed_data = cleaned_df
        return cleaned_df
    
    def get_duplicate_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Mendapatkan laporan detail tentang duplikat dalam data"""
        duplicate_detector = DuplicateDetector(similarity_threshold=0.85)
        report = duplicate_detector.get_duplicate_report(
            df, 
            name_col='nama_tempat', 
            location_col='lokasi'
        )
        return report
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validasi kualitas data yang komprehensif"""
        quality_report = {
            'data_quality_score': 0.0,
            'issues': [],
            'recommendations': [],
            'statistics': {}
        }
        
        total_rows = len(df)
        if total_rows == 0:
            quality_report['issues'].append('Dataset kosong')
            return quality_report
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        missing_percentage = (missing_counts / total_rows * 100).round(2)
        
        for col, missing_pct in missing_percentage.items():
            if missing_pct > 10:
                quality_report['issues'].append(f'Kolom {col} memiliki {missing_pct}% data yang hilang')
                quality_report['recommendations'].append(f'Pertimbangkan untuk mengisi atau menghapus data yang hilang di kolom {col}')
        
        # Check for potential duplicates
        duplicate_report = self.get_duplicate_report(df)
        if duplicate_report['total_potential_duplicates'] > 0:
            quality_report['issues'].append(f'Ditemukan {duplicate_report["total_potential_duplicates"]} pasangan data yang berpotensi duplikat')
            quality_report['recommendations'].append('Gunakan fungsi pembersihan data untuk menghapus duplikat')
        
        # Check data ranges
        if 'harga' in df.columns:
            harga_stats = df['harga'].describe()
            if harga_stats['min'] <= 0:
                quality_report['issues'].append('Ditemukan harga yang tidak valid (≤ 0)')
            if harga_stats['max'] > 1000000:  # 1 juta
                quality_report['issues'].append('Ditemukan harga yang sangat tinggi (> 1 juta)')
        
        if 'rating' in df.columns:
            rating_out_of_range = ((df['rating'] < 1) | (df['rating'] > 5)).sum()
            if rating_out_of_range > 0:
                quality_report['issues'].append(f'{rating_out_of_range} rating di luar rentang 1-5')
        
        if 'jarak' in df.columns:
            jarak_negative = (df['jarak'] < 0).sum()
            if jarak_negative > 0:
                quality_report['issues'].append(f'{jarak_negative} jarak bernilai negatif')
        
        # Calculate quality score
        total_issues = len(quality_report['issues'])
        quality_score = max(0, 100 - (total_issues * 10))  # Deduct 10 points per issue
        quality_report['data_quality_score'] = quality_score
        
        # Add statistics
        quality_report['statistics'] = {
            'total_rows': total_rows,
            'total_columns': len(df.columns),
            'missing_data_percentage': missing_percentage.to_dict(),
            'potential_duplicates': duplicate_report['total_potential_duplicates']
        }
        
        return quality_report
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Mendapatkan ringkasan statistik data"""
        summary = {
            'basic_info': {
                'total_restaurants': len(df),
                'unique_types': df['tipe_tempat'].nunique() if 'tipe_tempat' in df.columns else 0,
                'date_range': 'N/A'  # Bisa ditambahkan jika ada kolom tanggal
            },
            'price_stats': {},
            'rating_stats': {},
            'distance_stats': {},
            'type_distribution': {}
        }
        
        # Price statistics
        if 'harga' in df.columns:
            summary['price_stats'] = {
                'min': float(df['harga'].min()),
                'max': float(df['harga'].max()),
                'mean': float(df['harga'].mean()),
                'median': float(df['harga'].median()),
                'std': float(df['harga'].std())
            }
        
        # Rating statistics
        if 'rating' in df.columns:
            summary['rating_stats'] = {
                'min': float(df['rating'].min()),
                'max': float(df['rating'].max()),
                'mean': float(df['rating'].mean()),
                'median': float(df['rating'].median()),
                'std': float(df['rating'].std())
            }
        
        # Distance statistics
        if 'jarak' in df.columns:
            summary['distance_stats'] = {
                'min': float(df['jarak'].min()),
                'max': float(df['jarak'].max()),
                'mean': float(df['jarak'].mean()),
                'median': float(df['jarak'].median()),
                'std': float(df['jarak'].std())
            }
        
        # Type distribution
        if 'tipe_tempat' in df.columns:
            summary['type_distribution'] = df['tipe_tempat'].value_counts().to_dict()
        
        return summary
    
    def filter_data(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Filter data berdasarkan kriteria tertentu"""
        filtered_df = df.copy()
        
        # Filter by price range
        if 'min_harga' in filters and filters['min_harga'] is not None:
            filtered_df = filtered_df[filtered_df['harga'] >= filters['min_harga']]
        
        if 'max_harga' in filters and filters['max_harga'] is not None:
            filtered_df = filtered_df[filtered_df['harga'] <= filters['max_harga']]
        
        # Filter by rating range
        if 'min_rating' in filters and filters['min_rating'] is not None:
            filtered_df = filtered_df[filtered_df['rating'] >= filters['min_rating']]
        
        if 'max_rating' in filters and filters['max_rating'] is not None:
            filtered_df = filtered_df[filtered_df['rating'] <= filters['max_rating']]
        
        # Filter by distance range
        if 'min_jarak' in filters and filters['min_jarak'] is not None:
            filtered_df = filtered_df[filtered_df['jarak'] >= filters['min_jarak']]
        
        if 'max_jarak' in filters and filters['max_jarak'] is not None:
            filtered_df = filtered_df[filtered_df['jarak'] <= filters['max_jarak']]
        
        # Filter by restaurant type
        if 'tipe_tempat' in filters and filters['tipe_tempat']:
            if isinstance(filters['tipe_tempat'], list):
                filtered_df = filtered_df[filtered_df['tipe_tempat'].isin(filters['tipe_tempat'])]
            else:
                filtered_df = filtered_df[
                    filtered_df['tipe_tempat'].str.contains(
                        filters['tipe_tempat'], case=False, na=False
                    )
                ]
        
        return filtered_df
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Mendapatkan statistik ringkasan data yang sudah diproses"""
        if self.processed_data is None:
            return {
                'total_restaurants': 0,
                'avg_price': 0,
                'avg_rating': 0,
                'avg_distance': 0
            }
        
        return {
            'total_restaurants': len(self.processed_data),
            'avg_price': float(self.processed_data['harga'].mean()) if 'harga' in self.processed_data.columns else 0,
            'avg_rating': float(self.processed_data['rating'].mean()) if 'rating' in self.processed_data.columns else 0,
            'avg_distance': float(self.processed_data['jarak'].mean()) if 'jarak' in self.processed_data.columns else 0
        }
    
    def set_clustered_data(self, clustered_data: pd.DataFrame) -> None:
        """Set data yang sudah di-cluster"""
        self.clustered_data = clustered_data
    
    def get_filtered_data(self, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Mendapatkan data yang sudah difilter"""
        # Prioritas: gunakan clustered_data jika tersedia, jika tidak gunakan processed_data
        data_source = getattr(self, 'clustered_data', None)
        if data_source is None:
            data_source = self.processed_data
            
        if data_source is None:
            return pd.DataFrame()
        
        if filters is None:
            return data_source.copy()
        
        return self.filter_data(data_source, filters)
    
    def export_data(self, df: pd.DataFrame, output_path: str, format: str = 'csv') -> None:
        """Export data ke file"""
        try:
            if format.lower() == 'csv':
                df.to_csv(output_path, index=False)
            elif format.lower() == 'json':
                df.to_json(output_path, orient='records', indent=2)
            elif format.lower() == 'excel':
                df.to_excel(output_path, index=False)
            else:
                raise ValueError(f"Format tidak didukung: {format}")
                
            print(f"Data berhasil diekspor ke: {output_path}")
        except Exception as e:
            raise ValueError(f"Error saat ekspor data: {str(e)}")
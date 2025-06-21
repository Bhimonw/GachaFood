import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import os
from pathlib import Path

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
        
        # Remove duplicates based on nama_tempat and lokasi
        cleaned_df = cleaned_df.drop_duplicates(subset=['nama_tempat', 'lokasi'], keep='first')
        
        final_rows = len(cleaned_df)
        removed_rows = initial_rows - final_rows
        
        if removed_rows > 0:
            print(f"Removed {removed_rows} rows during cleaning process")
        
        self.processed_data = cleaned_df
        return cleaned_df
    
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
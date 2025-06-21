import pandas as pd
import re
from typing import List, Dict, Tuple
from difflib import SequenceMatcher

class DuplicateDetector:
    """Advanced duplicate detection for restaurant data"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        
    def normalize_text(self, text: str) -> str:
        """Normalize text for better comparison"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common punctuation
        text = re.sub(r'[.,;:!?()\[\]{}"\'-]', '', text)
        
        # Remove common words that don't add meaning
        common_words = ['restoran', 'warung', 'cafe', 'kedai', 'rumah makan', 'rm', 'depot']
        for word in common_words:
            text = re.sub(rf'\b{word}\b', '', text)
        
        # Remove extra spaces again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        normalized1 = self.normalize_text(text1)
        normalized2 = self.normalize_text(text2)
        
        if not normalized1 or not normalized2:
            return 0.0
        
        return SequenceMatcher(None, normalized1, normalized2).ratio()
    
    def find_potential_duplicates(self, df: pd.DataFrame, 
                                name_col: str = 'nama_tempat', 
                                location_col: str = 'lokasi') -> List[Tuple[int, int, float, float]]:
        """Find potential duplicates based on name and location similarity"""
        potential_duplicates = []
        
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                name_sim = self.calculate_similarity(
                    df.iloc[i][name_col], 
                    df.iloc[j][name_col]
                )
                
                location_sim = self.calculate_similarity(
                    df.iloc[i][location_col], 
                    df.iloc[j][location_col]
                )
                
                # Consider as potential duplicate if either name or location is very similar
                # and the other is at least moderately similar
                if ((name_sim >= self.similarity_threshold and location_sim >= 0.6) or
                    (location_sim >= self.similarity_threshold and name_sim >= 0.6) or
                    (name_sim >= 0.9 and location_sim >= 0.4)):
                    
                    potential_duplicates.append((i, j, name_sim, location_sim))
        
        return potential_duplicates
    
    def remove_duplicates_advanced(self, df: pd.DataFrame, 
                                 name_col: str = 'nama_tempat', 
                                 location_col: str = 'lokasi',
                                 rating_col: str = 'rating') -> pd.DataFrame:
        """Remove duplicates using advanced similarity matching"""
        if len(df) == 0:
            return df
        
        # First, do basic exact duplicate removal after normalization
        df_copy = df.copy()
        df_copy['nama_normalized'] = df_copy[name_col].apply(self.normalize_text)
        df_copy['lokasi_normalized'] = df_copy[location_col].apply(self.normalize_text)
        
        # Remove exact duplicates
        exact_duplicates = df_copy.duplicated(subset=['nama_normalized', 'lokasi_normalized']).sum()
        df_copy = df_copy.drop_duplicates(subset=['nama_normalized', 'lokasi_normalized'], keep='first')
        
        # Find potential fuzzy duplicates
        potential_duplicates = self.find_potential_duplicates(df_copy, name_col, location_col)
        
        # Decide which duplicates to remove
        indices_to_remove = set()
        
        for i, j, name_sim, location_sim in potential_duplicates:
            if i in indices_to_remove or j in indices_to_remove:
                continue
            
            # Keep the one with higher rating, or first one if ratings are equal
            rating_i = df_copy.iloc[i][rating_col] if rating_col in df_copy.columns else 0
            rating_j = df_copy.iloc[j][rating_col] if rating_col in df_copy.columns else 0
            
            if rating_j > rating_i:
                indices_to_remove.add(i)
            else:
                indices_to_remove.add(j)
        
        # Remove the identified duplicates
        df_final = df_copy.drop(df_copy.index[list(indices_to_remove)])
        
        # Drop temporary columns
        df_final = df_final.drop(columns=['nama_normalized', 'lokasi_normalized'])
        
        fuzzy_duplicates = len(indices_to_remove)
        total_removed = exact_duplicates + fuzzy_duplicates
        
        print(f"Advanced duplicate removal summary:")
        print(f"  - Initial rows: {len(df)}")
        print(f"  - Exact duplicates removed: {exact_duplicates}")
        print(f"  - Fuzzy duplicates removed: {fuzzy_duplicates}")
        print(f"  - Total duplicates removed: {total_removed}")
        print(f"  - Final rows: {len(df_final)}")
        
        return df_final
    
    def get_duplicate_report(self, df: pd.DataFrame, 
                           name_col: str = 'nama_tempat', 
                           location_col: str = 'lokasi') -> Dict:
        """Generate a detailed report of potential duplicates"""
        potential_duplicates = self.find_potential_duplicates(df, name_col, location_col)
        
        report = {
            'total_potential_duplicates': len(potential_duplicates),
            'duplicate_pairs': []
        }
        
        for i, j, name_sim, location_sim in potential_duplicates:
            pair_info = {
                'index_1': i,
                'index_2': j,
                'name_1': df.iloc[i][name_col],
                'name_2': df.iloc[j][name_col],
                'location_1': df.iloc[i][location_col],
                'location_2': df.iloc[j][location_col],
                'name_similarity': round(name_sim, 3),
                'location_similarity': round(location_sim, 3)
            }
            report['duplicate_pairs'].append(pair_info)
        
        return report
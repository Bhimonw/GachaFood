import re
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

def format_currency(amount: Union[int, float], currency: str = 'IDR') -> str:
    """Format currency amount with proper formatting"""
    try:
        if pd.isna(amount) or amount is None:
            return 'N/A'
        
        amount = float(amount)
        
        if currency == 'IDR':
            # Format Indonesian Rupiah
            if amount >= 1000000:
                return f"Rp {amount/1000000:.1f}M"
            elif amount >= 1000:
                return f"Rp {amount/1000:.0f}K"
            else:
                return f"Rp {amount:,.0f}"
        else:
            return f"{currency} {amount:,.2f}"
    
    except (ValueError, TypeError):
        return 'N/A'

def format_distance(distance: Union[int, float], unit: str = 'km') -> str:
    """Format distance with appropriate unit"""
    try:
        if pd.isna(distance) or distance is None:
            return 'N/A'
        
        distance = float(distance)
        
        if unit == 'km':
            if distance < 1:
                return f"{distance*1000:.0f} m"
            else:
                return f"{distance:.1f} km"
        else:
            return f"{distance:.1f} {unit}"
    
    except (ValueError, TypeError):
        return 'N/A'

def format_rating(rating: Union[int, float], max_rating: int = 5) -> str:
    """Format rating with stars or numeric display"""
    try:
        if pd.isna(rating) or rating is None:
            return 'N/A'
        
        rating = float(rating)
        
        # Clamp rating to valid range
        rating = max(0, min(rating, max_rating))
        
        # Create star representation
        full_stars = int(rating)
        half_star = 1 if (rating - full_stars) >= 0.5 else 0
        empty_stars = max_rating - full_stars - half_star
        
        stars = '★' * full_stars + '☆' * half_star + '☆' * empty_stars
        
        return f"{rating:.1f}/5 {stars}"
    
    except (ValueError, TypeError):
        return 'N/A'

def validate_input(value: Any, input_type: str, **kwargs) -> tuple[bool, Any, str]:
    """Validate input based on type and constraints"""
    try:
        if value is None or (isinstance(value, str) and value.strip() == ''):
            if kwargs.get('required', False):
                return False, None, "Value is required"
            return True, None, ""
        
        if input_type == 'int':
            try:
                validated_value = int(value)
                min_val = kwargs.get('min_value')
                max_val = kwargs.get('max_value')
                
                if min_val is not None and validated_value < min_val:
                    return False, None, f"Value must be at least {min_val}"
                if max_val is not None and validated_value > max_val:
                    return False, None, f"Value must be at most {max_val}"
                
                return True, validated_value, ""
            except ValueError:
                return False, None, "Invalid integer value"
        
        elif input_type == 'float':
            try:
                validated_value = float(value)
                min_val = kwargs.get('min_value')
                max_val = kwargs.get('max_value')
                
                if min_val is not None and validated_value < min_val:
                    return False, None, f"Value must be at least {min_val}"
                if max_val is not None and validated_value > max_val:
                    return False, None, f"Value must be at most {max_val}"
                
                return True, validated_value, ""
            except ValueError:
                return False, None, "Invalid float value"
        
        elif input_type == 'string':
            validated_value = str(value).strip()
            min_length = kwargs.get('min_length', 0)
            max_length = kwargs.get('max_length')
            pattern = kwargs.get('pattern')
            
            if len(validated_value) < min_length:
                return False, None, f"String must be at least {min_length} characters"
            if max_length and len(validated_value) > max_length:
                return False, None, f"String must be at most {max_length} characters"
            if pattern and not re.match(pattern, validated_value):
                return False, None, "String format is invalid"
            
            return True, validated_value, ""
        
        elif input_type == 'list':
            if not isinstance(value, list):
                return False, None, "Value must be a list"
            
            min_length = kwargs.get('min_length', 0)
            max_length = kwargs.get('max_length')
            
            if len(value) < min_length:
                return False, None, f"List must have at least {min_length} items"
            if max_length and len(value) > max_length:
                return False, None, f"List must have at most {max_length} items"
            
            return True, value, ""
        
        else:
            return False, None, f"Unknown input type: {input_type}"
    
    except Exception as e:
        return False, None, f"Validation error: {str(e)}"

def clean_text(text: str) -> str:
    """Clean and normalize text data"""
    if pd.isna(text) or text is None:
        return ''
    
    # Convert to string and strip whitespace
    text = str(text).strip()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?()-]', '', text)
    
    return text

def safe_divide(numerator: Union[int, float], denominator: Union[int, float], default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division by zero"""
    try:
        if denominator == 0:
            return default
        return float(numerator) / float(denominator)
    except (ValueError, TypeError):
        return default

def calculate_percentile(data: List[Union[int, float]], percentile: float) -> float:
    """Calculate percentile of a list of numbers"""
    try:
        if not data:
            return 0.0
        
        data_array = np.array([x for x in data if not pd.isna(x)])
        if len(data_array) == 0:
            return 0.0
        
        return float(np.percentile(data_array, percentile))
    except Exception:
        return 0.0

def normalize_score(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float]) -> float:
    """Normalize a value to 0-1 range"""
    try:
        if pd.isna(value) or pd.isna(min_val) or pd.isna(max_val):
            return 0.0
        
        if max_val == min_val:
            return 0.5  # If no variation, return middle value
        
        normalized = (float(value) - float(min_val)) / (float(max_val) - float(min_val))
        return max(0.0, min(1.0, normalized))  # Clamp to [0, 1]
    except (ValueError, TypeError):
        return 0.0

def get_price_category(price: Union[int, float]) -> str:
    """Categorize price into budget categories"""
    try:
        if pd.isna(price) or price is None:
            return 'Unknown'
        
        price = float(price)
        
        if price <= 25000:
            return 'Budget'
        elif price <= 50000:
            return 'Moderate'
        elif price <= 100000:
            return 'Expensive'
        else:
            return 'Premium'
    except (ValueError, TypeError):
        return 'Unknown'

def get_distance_category(distance: Union[int, float]) -> str:
    """Categorize distance into proximity categories"""
    try:
        if pd.isna(distance) or distance is None:
            return 'Unknown'
        
        distance = float(distance)
        
        if distance <= 1:
            return 'Very Close'
        elif distance <= 3:
            return 'Close'
        elif distance <= 5:
            return 'Moderate'
        elif distance <= 10:
            return 'Far'
        else:
            return 'Very Far'
    except (ValueError, TypeError):
        return 'Unknown'

def get_rating_category(rating: Union[int, float]) -> str:
    """Categorize rating into quality categories"""
    try:
        if pd.isna(rating) or rating is None:
            return 'Unknown'
        
        rating = float(rating)
        
        if rating >= 4.5:
            return 'Excellent'
        elif rating >= 4.0:
            return 'Very Good'
        elif rating >= 3.5:
            return 'Good'
        elif rating >= 3.0:
            return 'Average'
        elif rating >= 2.0:
            return 'Below Average'
        else:
            return 'Poor'
    except (ValueError, TypeError):
        return 'Unknown'

def paginate_data(data: List[Any], page: int, per_page: int) -> Dict[str, Any]:
    """Paginate a list of data"""
    try:
        total = len(data)
        start = (page - 1) * per_page
        end = start + per_page
        
        # Ensure valid page bounds
        if start < 0:
            start = 0
        if end > total:
            end = total
        
        items = data[start:end]
        
        return {
            'items': items,
            'page': page,
            'per_page': per_page,
            'total': total,
            'pages': (total + per_page - 1) // per_page if per_page > 0 else 1,
            'has_prev': page > 1,
            'has_next': end < total,
            'prev_num': page - 1 if page > 1 else None,
            'next_num': page + 1 if end < total else None
        }
    except Exception:
        return {
            'items': [],
            'page': 1,
            'per_page': per_page,
            'total': 0,
            'pages': 1,
            'has_prev': False,
            'has_next': False,
            'prev_num': None,
            'next_num': None
        }

def create_response(success: bool, data: Any = None, message: str = '', errors: List[str] = None) -> Dict[str, Any]:
    """Create standardized API response"""
    response = {
        'success': success,
        'message': message
    }
    
    if data is not None:
        response['data'] = data
    
    if errors:
        response['errors'] = errors
    
    return response

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """Handle missing values in DataFrame"""
    df_clean = df.copy()
    
    if strategy == 'drop':
        df_clean = df_clean.dropna()
    elif strategy == 'fill_mean':
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_columns] = df_clean[numeric_columns].fillna(df_clean[numeric_columns].mean())
        
        # Fill non-numeric with mode or default
        for col in df_clean.columns:
            if col not in numeric_columns:
                if df_clean[col].dtype == 'object':
                    mode_val = df_clean[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                    df_clean[col] = df_clean[col].fillna(fill_val)
    elif strategy == 'fill_median':
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_columns] = df_clean[numeric_columns].fillna(df_clean[numeric_columns].median())
    
    return df_clean
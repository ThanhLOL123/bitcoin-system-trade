import pandas as pd
import os
from datetime import datetime, timedelta

class FeatureStore:
    """Store and retrieve features using Parquet"""
    
    def __init__(self, base_path: str = 'feature_store'):
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def save_features(self, df: pd.DataFrame, date: str):
        """Save features to a Parquet file for a specific date"""
        file_path = os.path.join(self.base_path, f'{date}.parquet')
        df.to_parquet(file_path)

    def load_features(self, date: str) -> pd.DataFrame:
        """Load features from a Parquet file for a specific date"""
        file_path = os.path.join(self.base_path, f'{date}.parquet')
        try:
            return pd.read_parquet(file_path)
        except FileNotFoundError:
            return pd.DataFrame()

    def load_features_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load features for a range of dates"""
        all_features = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')

        while current_date <= end_date_dt:
            date_str = current_date.strftime('%Y-%m-%d')
            df = self.load_features(date_str)
            if not df.empty:
                all_features.append(df)
            current_date += timedelta(days=1)
        
        if all_features:
            return pd.concat(all_features).reset_index(drop=True)
        return pd.DataFrame()

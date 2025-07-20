import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataUtils:
    """Utilities for data preprocessing"""
    
    @staticmethod
    def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
        """Split data into training and testing sets"""
        X = df.drop(columns=['target'])
        y = df['target']
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    @staticmethod
    def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """Scale features using StandardScaler"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    @staticmethod
    def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values by filling with the mean"""
        return df.fillna(df.mean())
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.xgboost
from datetime import datetime, timedelta

from ..utils.data_utils import DataUtils
from ..feature_engineering.feature_store import FeatureStore
from ..models.lstm_model import LSTMModel
from ..models.transformer_model import TransformerModel
from ..validation.validator import ModelValidator

class Trainer:
    """Train machine learning models"""
    
    def __init__(self):
        self.data_utils = DataUtils()
        self.feature_store = FeatureStore()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_validator = ModelValidator()

    def _prepare_data_for_pytorch(self, df: pd.DataFrame, sequence_length: int):
        # Assuming 'target' is the last column and features are the rest
        features = df.drop(columns=['target']).values
        targets = df['target'].values

        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:i+sequence_length])
            y.append(targets[i+sequence_length])
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        return TensorDataset(X, y)

    def train(self):
        """Train various machine learning models"""
        # Load data for a range of dates (e.g., last 30 days)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        features_df = self.feature_store.load_features_range(start_date, end_date)

        if features_df.empty:
            print("Feature store is empty for the given date range. Cannot train model.")
            return

        # For simplicity, we'll create a binary target variable
        features_df['target'] = (features_df['price_usd'].shift(-1) > features_df['price_usd']).astype(int)
        features_df = features_df.dropna()

        # Handle missing values before splitting and scaling
        features_df = self.data_utils.handle_missing_values(features_df)

        X = features_df.drop(columns=['target'])
        y = features_df['target']

        X_train, X_test, y_train, y_test = self.data_utils.split_data(features_df)
        X_train_scaled, X_test_scaled = self.data_utils.scale_features(X_train, X_test)

        # Train traditional ML models
        models_to_train = {
            'logistic_regression': LogisticRegression(max_iter=1000),
            'random_forest': RandomForestClassifier(),
            'mlp_classifier': MLPClassifier(max_iter=1000)
        }

        for name, model in models_to_train.items():
            with mlflow.start_run(run_name=f"{name}_training"):
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                metrics = self.model_validator.calculate_classification_metrics(y_test, y_pred)
                backtest_results = self.model_validator.run_backtest(y_pred, y_test)

                print(f"{name} accuracy: {metrics['accuracy']}")
                mlflow.log_metrics(metrics)
                mlflow.log_metrics(backtest_results)
                mlflow.sklearn.log_model(model, name)

        # Train XGBoost model
        with mlflow.start_run(run_name="xgboost_training"):
            dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
            dtest = xgb.DMatrix(X_test_scaled, label=y_test)

            def objective_xgb(trial):
                param = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
                    'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
                    'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
                    'subsample': trial.suggest_float('subsample', 0.2, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
                    'max_depth': trial.suggest_int('max_depth', 3, 9),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'eta': trial.suggest_loguniform('eta', 1e-8, 1.0),
                    'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
                    'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
                }
                bst = xgb.train(param, dtrain)
                preds = bst.predict(dtest)
                accuracy = ((preds > 0.5) == y_test).mean()
                return accuracy

            study_xgb = optuna.create_study(direction="maximize")
            study_xgb.optimize(objective_xgb, n_trials=10)
            best_xgb_params = study_xgb.best_params
            mlflow.log_params(best_xgb_params)

            best_xgb_model = xgb.train(best_xgb_params, dtrain)
            preds = best_xgb_model.predict(dtest)
            
            metrics = self.model_validator.calculate_classification_metrics(y_test, (preds > 0.5).astype(int))
            backtest_results = self.model_validator.run_backtest(preds, y_test)

            mlflow.log_metrics(metrics)
            mlflow.log_metrics(backtest_results)
            mlflow.xgboost.log_model(best_xgb_model, "xgboost_model")
            print(f"XGBoost accuracy: {metrics['accuracy']}")

        # Prepare data for PyTorch models (LSTM, Transformer)
        sequence_length = 10 # Example sequence length
        pytorch_dataset = self._prepare_data_for_pytorch(features_df, sequence_length)
        train_loader = DataLoader(pytorch_dataset, batch_size=32, shuffle=True)

        # Train LSTM model
        with mlflow.start_run(run_name="lstm_training"):
            input_size = X_train_scaled.shape[1]
            hidden_size = 50
            num_layers = 2
            output_size = 1

            lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(self.device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

            def objective_lstm(trial):
                hidden_size = trial.suggest_int('hidden_size', 32, 128)
                num_layers = trial.suggest_int('num_layers', 1, 3)
                lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)

                model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=lr)

                for epoch in range(5): # Train for a few epochs for optimization
                    for inputs, labels in train_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                
                # Evaluate on a small subset for optimization
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for inputs, labels in train_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = model(inputs)
                        predicted = (torch.sigmoid(outputs) > 0.5).float()
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    accuracy = correct / total
                return accuracy

            study_lstm = optuna.create_study(direction="maximize")
            study_lstm.optimize(objective_lstm, n_trials=5)
            best_lstm_params = study_lstm.best_params
            mlflow.log_params(best_lstm_params)

            # Train final LSTM model
            final_lstm_model = LSTMModel(input_size, best_lstm_params['hidden_size'], best_lstm_params['num_layers'], output_size).to(self.device)
            final_optimizer = optim.Adam(final_lstm_model.parameters(), lr=best_lstm_params['lr'])
            for epoch in range(10): # Train for more epochs for final model
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    final_optimizer.zero_grad()
                    outputs = final_lstm_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    final_optimizer.step()
            
            # Evaluate final LSTM model
            with torch.no_grad():
                all_preds = []
                all_labels = []
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = final_lstm_model(inputs)
                    all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
                metrics = self.model_validator.calculate_classification_metrics(np.array(all_labels), (np.array(all_preds) > 0.5).astype(int))
                backtest_results = self.model_validator.run_backtest(np.array(all_preds), np.array(all_labels))
                mlflow.log_metrics(metrics)
                mlflow.log_metrics(backtest_results)

            mlflow.pytorch.log_model(final_lstm_model, "lstm_model")
            print(f"LSTM training complete.")

        # Train Transformer model
        with mlflow.start_run(run_name="transformer_training"):
            input_size = X_train_scaled.shape[1]
            d_model = 64
            nhead = 4
            num_layers = 2
            output_size = 1

            transformer_model = TransformerModel(input_size, d_model, nhead, num_layers, output_size).to(self.device)
            criterion = nn.BCEWithWithLogitsLoss()
            optimizer = optim.Adam(transformer_model.parameters(), lr=0.001)

            def objective_transformer(trial):
                d_model = trial.suggest_int('d_model', 32, 128)
                nhead = trial.suggest_int('nhead', 2, 8)
                num_layers = trial.suggest_int('num_layers', 1, 3)
                lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)

                model = TransformerModel(input_size, d_model, nhead, num_layers, output_size).to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=lr)

                for epoch in range(5): # Train for a few epochs for optimization
                    for inputs, labels in train_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                
                # Evaluate on a small subset for optimization
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for inputs, labels in train_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = model(inputs)
                        predicted = (torch.sigmoid(outputs) > 0.5).float()
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    accuracy = correct / total
                return accuracy

            study_transformer = optuna.create_study(direction="maximize")
            study_transformer.optimize(objective_transformer, n_trials=5)
            best_transformer_params = study_transformer.best_params
            mlflow.log_params(best_transformer_params)

            # Train final Transformer model
            final_transformer_model = TransformerModel(input_size, best_transformer_params['d_model'], best_transformer_params['nhead'], best_transformer_params['num_layers'], output_size).to(self.device)
            final_optimizer = optim.Adam(final_transformer_model.parameters(), lr=best_transformer_params['lr'])
            for epoch in range(10): # Train for more epochs for final model
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    final_optimizer.zero_grad()
                    outputs = final_transformer_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    final_optimizer.step()
            
            # Evaluate final Transformer model
            with torch.no_grad():
                all_preds = []
                all_labels = []
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = final_transformer_model(inputs)
                    all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
                metrics = self.model_validator.calculate_classification_metrics(np.array(all_labels), (np.array(all_preds) > 0.5).astype(int))
                backtest_results = self.model_validator.run_backtest(np.array(all_preds), np.array(all_labels))
                mlflow.log_metrics(metrics)
                mlflow.log_metrics(backtest_results)

            mlflow.pytorch.log_model(final_transformer_model, "transformer_model")
            print(f"Transformer training complete.")

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
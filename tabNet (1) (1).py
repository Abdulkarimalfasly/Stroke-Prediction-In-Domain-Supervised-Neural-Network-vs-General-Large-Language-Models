import torch
from typing import Tuple, List, Dict, Any
from pytorch_tabnet.tab_model import TabNetClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

class DataPreparationError(Exception):
    """Custom exception for data preparation errors."""
    pass

def prepare_data(file_path: str) -> Tuple[np.ndarray, pd.Series, List[str]]:
    """
    Prepare and clean the dataset for TabNet modeling.
    
    Args:
        file_path: Path to the CSV file containing the dataset
        
    Returns:
        Tuple containing:
        - Scaled feature matrix (numpy array)
        - Target variable series
        - List of feature names
        
    Raises:
        DataPreparationError: If there are issues with data preparation
        FileNotFoundError: If the input file doesn't exist
    """
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        print("Reading data...")
        data = pd.read_csv(file_path)
        
        print("Handling missing values...")
        if 'bmi' not in data.columns:
            raise DataPreparationError("Required column 'bmi' not found in dataset")
        data['bmi'].fillna(data['bmi'].median(), inplace=True)
        
        categorical_mappings = {
            'gender': {'Male': 0, 'Female': 1, 'Other': 2},
            'ever_married': {'No': 0, 'Yes': 1},
            'work_type': {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4},
            'Residence_type': {'Urban': 0, 'Rural': 1},
            'smoking_status': {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3}
        }
        
        print("Converting categorical variables...")
        for column, mapping in categorical_mappings.items():
            if column not in data.columns:
                raise DataPreparationError(f"Required column '{column}' not found in dataset")
            # Handle unknown categories with a default value
            data[column] = data[column].map(mapping).fillna(-1)
        
        if 'id' in data.columns:
            data = data.drop('id', axis=1)
        
        if 'stroke' not in data.columns:
            raise DataPreparationError("Target variable 'stroke' not found in dataset")
        
        print("Checking for remaining missing values...")
        missing_cols = data.columns[data.isnull().any()].tolist()
        if missing_cols:
            raise DataPreparationError(f"Missing values found in columns: {missing_cols}")
        
        print("Preparing features...")
        X = data.drop('stroke', axis=1)
        y = data['stroke']
        
        print("Normalizing features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, list(X.columns)
    
    except Exception as e:
        raise DataPreparationError(f"Error during data preparation: {str(e)}")

def plot_feature_importance(
    feature_importances: np.ndarray,
    feature_names: List[str],
    save_path: str = 'plots/feature_importance.png'
) -> None:
    """
    Plot and save feature importance visualization.
    
    Args:
        feature_importances: Array of feature importance scores
        feature_names: List of feature names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
        
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Feature Importance (TabNet)')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    finally:
        plt.close()

def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: str = 'plots/roc_curve.png'
) -> float:
    """
    Plot ROC curve and calculate AUC.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save the plot
        
    Returns:
        ROC AUC score
    """
    plt.figure(figsize=(8, 6))
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return roc_auc
    finally:
        plt.close()

def save_model_summary(
    model: TabNetClassifier,
    train_acc: float,
    val_acc: float,
    val_auc: float,
    feature_names: List[str],
    save_path: str = 'model_summary.txt'
) -> None:
    """
    Save model summary and performance metrics.
    
    Args:
        model: Trained TabNet model
        train_acc: Training accuracy
        val_acc: Validation accuracy
        val_auc: Validation AUC score
        feature_names: List of feature names
        save_path: Path to save the summary
    """
    os.makedirs('output', exist_ok=True)
    
    with open(f'output/{save_path}', 'w') as f:
        f.write("TabNet Model Summary and Results\n")
        f.write("="* 50 + "\n\n")
        
        f.write("Model Parameters:\n")
        f.write("-"* 30 + "\n")
        f.write(f"n_d (width of decision prediction layer): {model.n_d}\n")
        f.write(f"n_steps (number of steps): {model.n_steps}\n")
        f.write(f"gamma (feature transformer sparsity): {model.gamma}\n")
        f.write(f"n_independent: {model.n_independent}\n")
        f.write(f"n_shared: {model.n_shared}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write("-"* 30 + "\n")
        f.write(f"Training Accuracy: {train_acc:.2f}%\n")
        f.write(f"Validation Accuracy: {val_acc:.2f}%\n")
        f.write(f"Validation AUC: {val_auc:.4f}\n\n")
        
        f.write("Feature Importance Ranking:\n")
        f.write("-"* 30 + "\n")
        importances = model.feature_importances_
        feature_imp = [(f, imp) for f, imp in zip(feature_names, importances)]
        feature_imp.sort(key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(feature_imp, 1):
            f.write(f"{i}. {feature}: {importance:.4f}\n")

def plot_metrics(history: Any, save_path: str = 'output/training_metrics.png') -> None:
    """
    Plot training metrics over epochs.
    
    Args:
        history: Training history object from TabNet
        save_path: Path to save the plot
    """
    try:
        history_data = getattr(history, 'history', {})
        
        print("\nAvailable metrics in history:")
        metrics = list(history_data.keys())
        print(metrics)
        
        if not history_data:
            print("Warning: No metrics found in history")
            return

        # Create a grid of subplots based on metrics
        n_metrics = 4  # We'll plot Loss, AUC, Accuracy, and Logloss
        plt.figure(figsize=(20, 12))
        
        # Plot Loss
        plt.subplot(2, 2, 1)
        losses = history_data.get('loss', [])
        if losses:
            epochs = range(1, len(losses) + 1)
            plt.plot(epochs, losses, label='Loss', color='black')
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

        # Plot AUC
        plt.subplot(2, 2, 2)
        train_auc = history_data.get('train_auc', [])
        valid_auc = history_data.get('valid_auc', [])
        if train_auc and valid_auc:
            epochs = range(1, len(train_auc) + 1)
            plt.plot(epochs, train_auc, label='Train', color='blue')
            plt.plot(epochs, valid_auc, label='Validation', color='red')
            plt.title('AUC Scores')
            plt.xlabel('Epoch')
            plt.ylabel('AUC')
            plt.legend()
            plt.grid(True)

        # Plot Accuracy
        plt.subplot(2, 2, 3)
        train_acc = history_data.get('train_accuracy', [])
        valid_acc = history_data.get('valid_accuracy', [])
        if train_acc and valid_acc:
            epochs = range(1, len(train_acc) + 1)
            plt.plot(epochs, train_acc, label='Train', color='blue')
            plt.plot(epochs, valid_acc, label='Validation', color='red')
            plt.title('Accuracy Scores')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)

        # Plot Logloss
        plt.subplot(2, 2, 4)
        train_logloss = history_data.get('train_logloss', [])
        valid_logloss = history_data.get('valid_logloss', [])
        if train_logloss and valid_logloss:
            epochs = range(1, len(train_logloss) + 1)
            plt.plot(epochs, train_logloss, label='Train', color='blue')
            plt.plot(epochs, valid_logloss, label='Validation', color='red')
            plt.title('Log Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Log Loss')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    finally:
        plt.close()

def main() -> None:
    """Main execution function."""
    try:
        # Data preparation
        print("\nInitializing data preparation...")
        X_scaled, y, feature_names = prepare_data('healthcare-dataset-stroke-data.csv')
        
        # Split dataset
        print("Splitting dataset...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize TabNet classifier
        clf = TabNetClassifier(
            n_d=64,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            cat_idxs=[],
            cat_dims=[],
            cat_emb_dim=[],
            lambda_sparse=1e-3,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params=dict(
                mode="min",
                patience=5,
                min_lr=1e-5,
                factor=0.5
            ),
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            mask_type='sparsemax',
            verbose=1
        )
        
        # Train model
        print("\nStarting TabNet training...")
        clf.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_name=['train', 'valid'],
            eval_metric=['auc', 'accuracy', 'logloss'],
            max_epochs=50,
            patience=15,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        
        print("\nGenerating visualizations and saving results...")
        
        # Plot training metrics
        plot_metrics(clf.history)
        
        # Generate predictions
        y_pred = clf.predict(X_val)
        y_pred_proba = clf.predict_proba(X_val)
        
        # Calculate metrics using predictions
        train_pred = clf.predict(X_train)
        val_pred = clf.predict(X_val)
        train_acc = accuracy_score(y_train, train_pred) * 100
        val_acc = accuracy_score(y_val, val_pred) * 100
        
        # Plot and save visualizations
        val_auc = plot_roc_curve(y_val, y_pred_proba)
        plot_feature_importance(clf.feature_importances_, feature_names)
        
        # Save model summary
        save_model_summary(clf, train_acc, val_acc, val_auc, feature_names)
        
        # Print final results
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))
        
        print(f"\nModel Performance:")
        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        print(f"Validation AUC: {val_auc:.4f}")
        print(f"Best Validation AUC: {clf.best_cost:.4f}")
        
        # Save final model
        print("\nSaving model...")
        save_path = "tabnet_model.zip"
        clf.save_model(save_path)
        print(f"Model saved to: {save_path}")
        
        print("\nResults have been saved in:")
        print("- Plots: ./plots/")
        print("- Model Summary: ./model_summary.txt")
        print(f"- Model: ./{save_path}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
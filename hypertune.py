import json
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import toml
import torch
from mltrainer import ReportTypes, Trainer, TrainerSettings
from sklearn.metrics import classification_report, confusion_matrix

from src import metrics
from src.load_heart_data import get_heart_streamers
from src.model_ecg import SimpleCNN


def load_training_config(config_path: str = "config.toml") -> dict:
    """
    Load training configuration from TOML file.
    """
    try:
        config = toml.load(config_path)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using default values.")
        return get_default_training_config()
    except Exception as e:
        print(f"Error loading config: {e}. Using default values.")
        return get_default_training_config()


def get_default_training_config() -> dict:
    """
    Return default training configuration values.
    """
    return {
        'data': {
            'data_dir': str(Path("../data").resolve()),
            'dataset_name': "heart_big",
            'target_count': 15000,
            'batch_size': 32,
            'input_channels': 1
        },
        'training': {
            'max_epochs': 50,
            'patience': 10,
            'lr': 0.001,
            'weight_decay': 1e-4,
            'scheduler_factor': 0.1,
            'scheduler_patience': 5
        },
        'model': {
            'output_classes': 5,
            'dropout_rate': 0.3,
            'se_reduction': 16
        },
        'output': {
            'log_dir': "logs/CNN_Arrythmea",
            'confusion_matrix_file': "confusion_matrix.png",
            'results_file': "final_results.json",
            'class_names': ['N', 'S', 'V', 'F', 'Q'],
            'figure_dpi': 300
        },
        'hardware': {
            'device': "auto"  # "auto", "cpu", "cuda", "mps"
        }
    }


def get_device(device_config: str = "auto") -> torch.device:
    """
    Get the appropriate device based on configuration.
    """
    if device_config == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_config)


def train_simplecnn(config_path: str = "config.toml") -> Tuple[torch.nn.Module, Any]:
    """
    Trains a SimpleCNN model using configuration from TOML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        Tuple[torch.nn.Module, Any]: Trained model and validation streamer
    """
    # Load configuration
    full_config = load_training_config(config_path)
    
    # Extract config sections for easier access
    data_config = full_config['data']
    training_config = full_config['training']
    model_config = full_config['model']
    hardware_config = full_config['hardware']
    
    # Create data configuration for the data loader
    # (keeping the old format for compatibility with existing data loader)
    data_loader_config = {
        "data_dir": data_config['data_dir'],
        "dataset_name": data_config['dataset_name'],
        "target_count": data_config['target_count'],
        "batch_size": data_config['batch_size'],
        "input_channels": data_config['input_channels'],
        "output": model_config['output_classes'],  # For backward compatibility
        "dropout": model_config['dropout_rate'],   # For backward compatibility
        "lr": training_config['lr'],
        "weight_decay": training_config['weight_decay']
    }

    print("Loading data streamers...")
    trainstreamer, validstreamer = get_heart_streamers(data_loader_config)
    
    print("Initializing model...")
    model = SimpleCNN(config_path)  # Model now reads from same config file

    loss_fn = torch.nn.CrossEntropyLoss()

    metric_list = [
        metrics.Accuracy(),
        metrics.F1Score(average="macro"),
        metrics.Recall(average="macro"),
        metrics.Precision(average="macro"),
    ]

    device = get_device(hardware_config['device'])
    print(f"Using device: {device}")

    settings = TrainerSettings(
        epochs=training_config['max_epochs'],
        metrics=metric_list,
        logdir=Path(full_config['output']['log_dir']),
        train_steps=len(trainstreamer),
        valid_steps=len(validstreamer),
        reporttypes=[ReportTypes.TENSORBOARD],
        scheduler_kwargs={
            "factor": training_config['scheduler_factor'], 
            "patience": training_config['scheduler_patience']
        },
        optimizer_kwargs={
            "lr": training_config['lr'],
            "weight_decay": training_config['weight_decay']
        },
        earlystop_kwargs={"patience": training_config['patience']},
    )

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=torch.optim.AdamW,
        traindataloader=trainstreamer.stream(),
        validdataloader=validstreamer.stream(),
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        device=device,
    )

    print("Starting training...")
    trainer.loop()
    print("Training completed!")
    
    return model, validstreamer


def generate_confusion_matrix(model: torch.nn.Module, teststreamer: Any, 
                            config_path: str = "config.toml") -> Dict[str, Any]:
    """
    Generates and saves a confusion matrix and classification report.

    Args:
        model (torch.nn.Module): Trained model
        teststreamer (Any): Data streamer for test/validation data
        config_path (str): Path to configuration file

    Returns:
        Dict[str, Any]: Dictionary containing results and metadata
    """
    # Load configuration for output settings
    config = load_training_config(config_path)
    output_config = config['output']
    
    class_names = output_config['class_names']
    
    print("Generating predictions...")
    y_true = []
    y_pred = []

    testdata = teststreamer.stream()
    model.to("cpu")
    model.eval()

    with torch.no_grad():
        for _ in range(len(teststreamer)):
            try:
                X, y = next(testdata)
                yhat = model(X)
                yhat = yhat.argmax(dim=1)
                y_pred.append(yhat.cpu().tolist())
                y_true.append(y.cpu().tolist())
            except StopIteration:
                break

    # Flatten predictions and true labels
    yhat_flat = [x for y in y_pred for x in y]
    y_flat = [x for y in y_true for x in y]

    print("Computing confusion matrix...")
    cfm = confusion_matrix(y_flat, yhat_flat)
    cfm_normalized = cfm / np.sum(cfm, axis=1, keepdims=True)

    # Create visualization
    print("Creating confusion matrix visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cfm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')

    sns.heatmap(cfm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')

    plt.tight_layout()
    
    # Save with configured filename and DPI
    output_file = output_config['confusion_matrix_file']
    dpi = output_config['figure_dpi']
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Confusion matrix saved to {output_file}")
    plt.show()

    # Generate classification report
    print("Generating classification report...")
    report = classification_report(y_flat, yhat_flat, target_names=class_names,
                                   output_dict=True, zero_division=0)

    results = {
        "confusion_matrix": cfm.tolist(),
        "confusion_matrix_normalized": cfm_normalized.tolist(),
        "classification_report": report,
        "class_names": class_names,
        "total_samples": len(y_flat),
        "config_used": config  # Include the configuration used
    }

    # Save results with configured filename
    results_file = output_config['results_file']
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

    return results

if __name__ == "__main__":
    config_file = "config.toml"
    
    print("Starting training with configurable parameters...")
    model, teststreamer = train_simplecnn(config_file)
    
    print("Evaluating model performance...")
    results = generate_confusion_matrix(model, teststreamer, config_file)

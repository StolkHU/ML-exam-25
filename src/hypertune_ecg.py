import json
from pathlib import Path
from typing import Any, Dict, Tuple  # AS <added type hinting support>

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from mltrainer import (ReportTypes, Trainer,  # AS <custom training framework>
                       TrainerSettings)
from sklearn.metrics import classification_report, confusion_matrix

from load_heart_data import get_heart_streamers  # AS <custom data loader>
from src import metrics  # AS <custom metrics module>
from src.model import SimpleCNN  # AS <import of custom model>

MAX_EPOCHS = 50
PATIENCE = 10

def train_simplecnn() -> Tuple[torch.nn.Module, Any]:
    """
    Trains a SimpleCNN model using the provided configuration and returns the trained model and validation streamer.
    """
    config = {
        "data_dir": str(Path("../data").resolve()),  # AS <custom data path>
        "dataset_name": "heart_big",  # AS <custom dataset name>
        "target_count": 15000,  # AS <custom target count>
        "batch_size": 32,
        "input_channels": 1,
        "output": 5,
        "dropout": 0.3,  # AS <custom dropout parameter>
        "lr": 0.001,
        "weight_decay": 1e-4
    }

    trainstreamer, validstreamer = get_heart_streamers(config)  # AS <custom data streaming>
    model = SimpleCNN(config)  # AS <custom model initialization>

    loss_fn = torch.nn.CrossEntropyLoss()

    metric_list = [
        metrics.Accuracy(),  # AS <custom metric>
        metrics.F1Score(average="macro"),  # AS <custom metric>
        metrics.Recall(average="macro"),  # AS <custom metric>
        metrics.Precision(average="macro"),  # AS <custom metric>
    ]

    device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")  # AS <MPS support for Apple Silicon>

    settings = TrainerSettings(
        epochs=MAX_EPOCHS,
        metrics=metric_list,
        logdir=Path("logs/simplecnn"),  # AS <custom log directory>
        train_steps=len(trainstreamer),
        valid_steps=len(validstreamer),
        reporttypes=[ReportTypes.TENSORBOARD],  # AS <custom reporting>
        scheduler_kwargs={"factor": 0.1, "patience": 5},
        optimizer_kwargs={
            "lr": config["lr"],
            "weight_decay": config["weight_decay"]
        },
        earlystop_kwargs={"patience": PATIENCE},
    )

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=torch.optim.AdamW,  # AS <custom optimizer>
        traindataloader=trainstreamer.stream(),  # AS <custom dataloader>
        validdataloader=validstreamer.stream(),  # AS <custom dataloader>
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,  # AS <custom scheduler>
        device=device,
    )

    trainer.loop()
    return model, validstreamer

def generate_confusion_matrix(model: torch.nn.Module, teststreamer: Any) -> Dict[str, Any]:
    """
    Generates and saves a confusion matrix and classification report for the given model and test data.

    Args:
        model (torch.nn.Module): Trained model.
        teststreamer (Any): Data streamer for test/validation data.

    Returns:
        Dict[str, Any]: Dictionary containing confusion matrix, normalized matrix, classification report, and metadata.
    """
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

    yhat_flat = [x for y in y_pred for x in y]
    y_flat = [x for y in y_true for x in y]

    cfm = confusion_matrix(y_flat, yhat_flat)
    cfm_normalized = cfm / np.sum(cfm, axis=1, keepdims=True)

    class_names = ['N', 'S', 'V', 'F', 'Q']  # AS <custom class labels>

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
    plt.savefig('confusion_matrix_simplecnn_highdropout.png', dpi=300, bbox_inches='tight')  # AS <custom filename>
    plt.show()

    report = classification_report(y_flat, yhat_flat, target_names=class_names,
                                   output_dict=True, zero_division=0)

    results = {
        "confusion_matrix": cfm.tolist(),
        "confusion_matrix_normalized": cfm_normalized.tolist(),
        "classification_report": report,
        "class_names": class_names,
        "total_samples": len(y_flat)
    }

    with open("final_results_simplecnn.json", "w") as f:  # AS <custom output file>
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    model, teststreamer = train_simplecnn()
    results = generate_confusion_matrix(model, teststreamer)
    print(f"Final Accuracy: {results['classification_report']['accuracy']:.4f}")
    print(f"Final Recall (macro): {results['classification_report']['macro avg']['recall']:.4f}")


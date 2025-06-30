import torch
from pathlib import Path
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from src.model_final import SimpleCNN  # Assuming SimpleCNN is saved in src/model_1.py
from load_heart_data import get_heart_streamers
from src import metrics
from mltrainer import Trainer, TrainerSettings, ReportTypes

MAX_EPOCHS = 40
PATIENCE = 10

def train_simplecnn():
    config = {
        "data_dir": str(Path("../data").resolve()),
        "dataset_name": "heart_big",
        "target_count": 5000,
        "batch_size": 32,
        "input_channels": 1,
        "output": 5,
        "dropout": 0.3,
        "lr": 0.001,
        "weight_decay": 1e-4
    }

    trainstreamer, validstreamer = get_heart_streamers(config)
    model = SimpleCNN(config)

    loss_fn = torch.nn.CrossEntropyLoss()

    metric_list = [
        metrics.Accuracy(),
        metrics.F1Score(average="macro"),
        metrics.Recall(average="macro"),
        metrics.Precision(average="macro"),
    ]

    device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")

    settings = TrainerSettings(
        epochs=MAX_EPOCHS,
        metrics=metric_list,
        logdir=Path("logs/simplecnn"),
        train_steps=len(trainstreamer),
        valid_steps=len(validstreamer),
        reporttypes=[ReportTypes.TENSORBOARD],
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
        optimizer=torch.optim.AdamW,
        traindataloader=trainstreamer.stream(),
        validdataloader=validstreamer.stream(),
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        device=device,
    )

    trainer.loop()
    return model, validstreamer

def generate_confusion_matrix(model, teststreamer):
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

    yhat = [x for y in y_pred for x in y]
    y = [x for y in y_true for x in y]

    cfm = confusion_matrix(y, yhat)
    cfm_normalized = cfm / np.sum(cfm, axis=1, keepdims=True)

    class_names = ['N', 'S', 'V', 'F', 'Q']

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
    plt.savefig('confusion_matrix_simplecnn_highdropout.png', dpi=300, bbox_inches='tight')
    plt.show()

    report = classification_report(y, yhat, target_names=class_names,
                                   output_dict=True, zero_division=0)

    results = {
        "confusion_matrix": cfm.tolist(),
        "confusion_matrix_normalized": cfm_normalized.tolist(),
        "classification_report": report,
        "class_names": class_names,
        "total_samples": len(y)
    }

    with open("final_results_simplecnn.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    model, teststreamer = train_simplecnn()
    results = generate_confusion_matrix(model, teststreamer)
    print(f"Final Accuracy: {results['classification_report']['accuracy']:.4f}")
    print(f"Final Recall (macro): {results['classification_report']['macro avg']['recall']:.4f}")


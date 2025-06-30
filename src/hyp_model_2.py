from pathlib import Path
import torch
from loguru import logger
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from src.model_2 import ModularCNN
from load_heart_data import get_heart_streamers
from src import metrics
from mltrainer import Trainer, TrainerSettings, ReportTypes

MAX_EPOCHS = 40
PATIENCE = 15

def train_best_config():
    best_config = {
        "data_dir": str(Path("../data").resolve()),
        "dataset_name": "heart_big",
        "target_count": 20000,
        "batch_size": 32,
        "input_channels": 1,
        "output": 5,
        "num_conv_layers": 4,
        "layer_0_out_channels": 64,
        "layer_0_kernel_size": 7,
        "layer_0_pool": "max",
        "layer_1_out_channels": 64,
        "layer_1_kernel_size": 7,
        "layer_1_pool": "max",
        "layer_2_out_channels": 96,
        "layer_2_kernel_size": 3,
        "layer_2_pool": "avg",
        "layer_3_out_channels": 160,
        "layer_3_kernel_size": 3,
        "layer_3_pool": "none",
        "fc1_size": 256,
        "fc2_size": 256,
        "fc3_size": 96,
        "dropout": 0.42,
        "squeeze_excite": True,
        "label_smoothing": 0.0,
        "lr": 0.001,
        "weight_decay": 1e-4
    }

    conv_layers = []
    for i in range(4):
        conv_layers.append({
            "out_channels": best_config[f"layer_{i}_out_channels"],
            "kernel_size": best_config[f"layer_{i}_kernel_size"],
            "pool": best_config[f"layer_{i}_pool"]
        })

    model_config = {
        "input_channels": 1,
        "output": 5,
        "dropout": best_config["dropout"],
        "squeeze_excite": best_config["squeeze_excite"],
        "conv_layers": conv_layers,
        "fc1_size": best_config["fc1_size"],
        "fc2_size": best_config["fc2_size"],
        "fc3_size": best_config["fc3_size"]
    }

    trainstreamer, validstreamer = get_heart_streamers(best_config)
    model = ModularCNN(model_config)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=best_config["label_smoothing"])
    
    metric_list = [
        metrics.Accuracy(),
        metrics.F1Score(average="macro"),
        metrics.Recall(average="macro"),
        metrics.Precision(average="macro"),
    ]

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    settings = TrainerSettings(
        epochs=MAX_EPOCHS,
        metrics=metric_list,
        logdir=Path("logs/final_best_config"),
        train_steps=len(trainstreamer),
        valid_steps=len(validstreamer),
        reporttypes=[ReportTypes.TENSORBOARD],
        scheduler_kwargs={"factor": 0.2, "patience": 5},
        optimizer_kwargs={
            "lr": best_config["lr"], 
            "weight_decay": best_config["weight_decay"]
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
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
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
    
    with open("final_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    model, teststreamer = train_best_config()
    results = generate_confusion_matrix(model, teststreamer)
    
    logger.info(f"Final Accuracy: {results['classification_report']['accuracy']:.4f}")
    logger.info(f"Final Recall (macro): {results['classification_report']['macro avg']['recall']:.4f}")
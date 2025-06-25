from pathlib import Path
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import torch
from loguru import logger

from src.models import ModularCNN
from load_heart_data import get_heart_streamers
from src import metrics
from mltrainer import Trainer, TrainerSettings, ReportTypes

NUM_SAMPLES = 25
MAX_EPOCHS = 10

def train(config):
    # Dynamisch conv_layers genereren op basis van hyperparameters
    conv_layers = []
    for i in range(config["num_conv_layers"]):
        conv_layers.append({
            "out_channels": config[f"layer_{i}_out_channels"],
            "kernel_size": config[f"layer_{i}_kernel_size"],
            "pool": config.get(f"layer_{i}_pool", "none")
        })

    # Bouw het modelconfiguratieobject inclusief FC-laaggroottes
    model_config = {
        "input_channels": config["input_channels"],
        "output": config["output"],
        "dropout": config["dropout"],
        "squeeze_excite": config.get("squeeze_excite", False),
        "attention": config.get("attention", False),
        "skip_layers": config.get("skip_layers", []),
        "conv_layers": conv_layers,
        "fc1_size": config["fc1_size"],
        "fc2_size": config["fc2_size"],
        "fc3_size": config["fc3_size"]
    }

    trainstreamer, validstreamer = get_heart_streamers(config)
    model = ModularCNN(model_config)

    loss_fn = torch.nn.CrossEntropyLoss()
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

    logger.info(f"Using device: {device}")

    settings = TrainerSettings(
        epochs=MAX_EPOCHS,
        metrics=metric_list,
        logdir=Path("logs/heart1D"),
        train_steps=len(trainstreamer),
        valid_steps=len(validstreamer),
        reporttypes=[ReportTypes.RAY, ReportTypes.TENSORBOARD],
        scheduler_kwargs={"factor": 0.5, "patience": 3},  # of conditioneel op scheduler
        optimizer_kwargs={"lr": config["lr"], "weight_decay": config["weight_decay"]},
        earlystop_kwargs=None,
)


    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=torch.optim.Adam,
        traindataloader=trainstreamer.stream(),
        validdataloader=validstreamer.stream(),
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        device=device,
    )

    trainer.loop()

if __name__ == "__main__":
    import ray
    ray.init()

    config = {
        "data_dir": str(Path("../data").resolve()),
        "dataset_name": "heart_big",
        "target_count": 15000,
        "batch_size": 32,
        "input_channels": 1,
        "output": 5,
        "dropout": tune.uniform(0.1, 0.5),
        "squeeze_excite": tune.choice([True, False]),
        "attention": tune.choice([True, False]),
        "skip_layers": tune.choice([[], [1], [2], [1, 2]]),
        "num_conv_layers": tune.choice([4, 6, 8, 10]),
        "min_out_channels": tune.choice([16, 32]),
        "max_out_channels": tune.choice([128, 160]),
      "lr": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        "optimizer": tune.choice(["Adam", "SGD", "AdamW"]),
        "scheduler": tune.choice(["ReduceLROnPlateau", "StepLR", "CosineAnnealingLR"]),
        "kernel_sizes": [[3, 5, 7, 9, 11]],
        "pool_strategy": tune.choice(["none", "max", "avg"]),
        "fc1_size": tune.choice([256, 384, 512]),
        "fc2_size": tune.choice([128, 256, 384]),
        "fc3_size": tune.choice([64, 96, 128]),
    }

    search_alg = HyperOptSearch()
    scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        grace_period=5,
        reduction_factor=2,
        max_t=MAX_EPOCHS,
    )

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")
    reporter.add_metric_column("F1Score_macro")
    reporter.add_metric_column("Recall_macro")
    reporter.add_metric_column("Precision_macro")

    tune.run(
        train,
        config=config,
        metric="Recall_macro",
        mode="max",
        num_samples=NUM_SAMPLES,
        search_alg=search_alg,
        scheduler=scheduler,
        progress_reporter=reporter,
        storage_path=str(Path("logs/ray").resolve()),
        verbose=1,
    )

    ray.shutdown()


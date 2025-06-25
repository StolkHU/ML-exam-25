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

NUM_SAMPLES = 50
MAX_EPOCHS = 20

def train(config):
    """Training functie - simpel en direct"""
    # AS: Haal data streamers op met config parameters
    trainstreamer, validstreamer = get_heart_streamers(config)
    
    # AS: Maak model direct met config
    model = ModularCNN(config)

    # AS: Standaard loss en metrics
    loss_fn = torch.nn.CrossEntropyLoss()
    metric_list = [
        metrics.Accuracy(),
        metrics.F1Score(average="macro"),
        metrics.Recall(average="macro"),
        metrics.Precision(average="macro"),
    ]

    # AS: Device selectie
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # AS: Trainer settings - simpel gehouden
    settings = TrainerSettings(
        epochs=MAX_EPOCHS,
        metrics=metric_list,
        logdir=Path("logs/heart1D"),
        train_steps=len(trainstreamer),
        valid_steps=len(validstreamer),
        reporttypes=[ReportTypes.RAY],
        scheduler_kwargs={"factor": 0.5, "patience": 3},
        optimizer_kwargs={"lr": config["lr"], "weight_decay": config["weight_decay"]},
        earlystop_kwargs=None,
    )

    # AS: Trainer setup
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

    # AS: Zeer eenvoudige config - alleen essentiële parameters
    config = {
        # AS: Data parameters - zorg voor correcte pad
        "data_dir": str(Path("../data").resolve()),  # AS: Aangepast pad zonder ../
        "dataset_name": "heart_big", 
        "target_count": 15000,
        "batch_size": tune.choice([32, 64]),
        
        # Model basis - simpel houden
        "input_channels": 1,
        "output": 5,
        "dropout": tune.uniform(0.2, 0.5),
        
        # AS: Parameters die matchen met het nieuwe model
        "num_conv_layers": tune.choice([3, 4, 5]),  # Bepaalt welke conv layers gebruikt worden
        "base_channels": tune.choice([16, 32, 64]), # Basis aantal channels
        "kernel_size": tune.choice([3, 5, 7]),      # Kernel size voor alle conv layers
        
        # AS: Training parameters
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
    }

    # AS: Ray Tune setup
    search_alg = HyperOptSearch()
    scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        grace_period=3,
        reduction_factor=3,
        max_t=MAX_EPOCHS,
    )

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")
    reporter.add_metric_column("F1Score_macro")
    reporter.add_metric_column("Recall_macro")

    # AS: Ray tune run
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
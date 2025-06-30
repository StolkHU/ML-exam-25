from pathlib import Path
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import torch
from loguru import logger
import ray

from src.models import ModularCNN
from load_heart_data import get_heart_streamers
from src import metrics
from mltrainer import Trainer, TrainerSettings, ReportTypes

NUM_SAMPLES = 20 # Minder samples
MAX_EPOCHS = 15    # 3 echte epoch is genoeg voor eerste verkenning
STEPS_PER_EPOCH = 5

def train(config):
    """Training functie - simpel en direct"""
    # AS: Haal data streamers op met config parameters
    trainstreamer, validstreamer = get_heart_streamers(config)

    loss_fn = torch.nn.CrossEntropyLoss()
    
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
    
    # AS: Bereken steps per "mini-epoch"
    steps_per_mini_epoch = len(trainstreamer) // STEPS_PER_EPOCH
    logger.info(f"Steps per mini-epoch: {steps_per_mini_epoch}")

    # AS: Trainer settings
    settings = TrainerSettings(
        epochs=MAX_EPOCHS,
        metrics=metric_list,
        logdir=Path("logs/heart1D"),
        train_steps=steps_per_mini_epoch,  
        valid_steps=len(validstreamer),
        reporttypes=[ReportTypes.RAY], 
        scheduler_kwargs={"factor": 0.5, "patience": 3},
        optimizer_kwargs={"lr": config["lr"], "weight_decay": config["weight_decay"]},
        earlystop_kwargs={"patience" : 4, "verbose" : True, "delta": 0.005, "save" : True} ,  ## Early stopping erbij!
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
    # Limiteer Ray geheugen gebruik
    ray.init(
        object_store_memory=2_000_000_000,  # 2GB object store
        _memory=4_000_000_000,  # 4GB totaal voor Ray
    )

    # AS: HyperOpt search - focus op klein vs medium modellen
    config = {
        # Data parameters 
        "data_dir": str(Path("../data").resolve()),
        "dataset_name": "heart_big", 
        "target_count": 5000,  
        "batch_size": tune.choice([32]),  
        
        # Model parameters - klein vs medium
        "input_channels": 1, 
        "output": 5,  
        "dropout": tune.uniform(0.2, 0.4),  # Meer dropout ===
        "num_conv_layers": tune.choice([4, 5]), 
        "base_channels": tune.choice([32, 64]),  # Kleinere modellen
        "kernel_size": tune.choice([5, 7]),  
        
        # Architecture flags - deze houden we  
        "use_skip": tune.choice([1]),  # Belangrijk voor diepe nets
        "use_attention": tune.choice([1]),  # Kan helpen bij complexiteit
        
        # FC layer ratios - ook extremen
        "fc1_size": tune.choice([512, 256]),
        "fc2_size": tune.choice([128, 256]),
        "fc3_size": tune.choice([64, 96]),
        
        # Training parameters
        "lr": tune.loguniform(1e-3, 3e-3),  # Hogere LR voor sneller trainen
        "weight_decay": 1e-4,  # Fix dit
    }
    
    # HyperOpt search
    search_alg = HyperOptSearch()
    
    scheduler = None  # Was AsyncHyperBandScheduler

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")
    reporter.add_metric_column("F1Score_macro")
    reporter.add_metric_column("Recall_macro")

    # AS: Ray tune run met HyperOpt
    analysis = tune.run(
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
        max_concurrent_trials=3,  # Terug naar 3 met kleinere modellen
        resources_per_trial={
            "cpu": 1,
            "gpu": 0
        },
        raise_on_failed_trial=False  # Continue bij failures
    )
    
    # Print beste config direct
    best_config = analysis.get_best_config(metric="Recall_macro", mode="max")
    print("\n" + "="*60)
    print("BESTE CONFIGURATIE:")
    print("="*60)
    for key, value in best_config.items():
        if not key.startswith('data_'):
            print(f"{key}: {value}")
    print(f"\nBeste Recall: {analysis.best_result['Recall_macro']:.4f}")
    
    # Analyses
    df = analysis.dataframe()
    df.to_csv("ray_results.csv", index=False)
    
    # Sla op
    import json
    with open("best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

    ray.shutdown()
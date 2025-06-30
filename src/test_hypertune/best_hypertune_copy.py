
# hyperparameter_optimization.py - Aangepaste optimalisatie
from pathlib import Path
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import torch
from loguru import logger
import json

from src.best_model_copy import ModularCNN
from load_heart_data import get_heart_streamers
from src import metrics
from mltrainer import Trainer, TrainerSettings, ReportTypes

NUM_SAMPLES = 20
MAX_EPOCHS = 15

def train(config):
    """Training function aangepast volgens jouw wensen"""
    # Generate conv_layers from hyperparameters
    conv_layers = []
    for i in range(config["num_conv_layers"]):
        conv_layers.append({
            "out_channels": config[f"layer_{i}_out_channels"],
            "kernel_size": config[f"layer_{i}_kernel_size"],
            "pool": config.get(f"layer_{i}_pool", "none")
        })

    # Model configuration - skip_layers en attention weggehaald
    model_config = {
        "input_channels": config["input_channels"],
        "output": config["output"],
        "dropout": config["dropout"],
        "squeeze_excite": config.get("squeeze_excite", False),
        "conv_layers": conv_layers,
        "fc1_size": config["fc1_size"],
        "fc2_size": config["fc2_size"],
        "fc3_size": config["fc3_size"]
    }

    trainstreamer, validstreamer = get_heart_streamers(config)
    model = ModularCNN(model_config)

    # Eenvoudige loss function (label smoothing eruit)
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

    # Training settings - factor naar 0.1, normale patience
    settings = TrainerSettings(
        epochs=MAX_EPOCHS,
        metrics=metric_list,
        logdir=Path("logs/heart1D_improved"),
        train_steps=len(trainstreamer),
        valid_steps=len(validstreamer),
        reporttypes=[ReportTypes.RAY],
        scheduler_kwargs={"factor": 0.1, "patience": 3},  # Aangepast zoals gevraagd
        optimizer_kwargs={
            "lr": config["lr"], 
            "weight_decay": config["weight_decay"]
        },
        earlystop_kwargs=None,
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

if __name__ == "__main__":
    import ray
    ray.init()

    # Aangepaste config volgens jouw wensen
    config = {
        # Data parameters
        "data_dir": str(Path("../data").resolve()),
        "dataset_name": "heart_big",
        "target_count": 15000,
        "batch_size": 32,
        
        # Model architecture
        "input_channels": 1,
        "output": 5,
        "num_conv_layers": 4,
        
        # Layer 0 - Veel grotere kernel sizes voor ECG (rond 11-15)
        "layer_0_out_channels": tune.choice([64, 80, 96]),
        "layer_0_kernel_size": tune.choice([11, 13, 15]),  # Veel groter voor ECG patronen
        "layer_0_pool": "max",
        
        # Layer 1 - Nog groter (rond 13-17)
        "layer_1_out_channels": tune.choice([80, 96, 112]),
        "layer_1_kernel_size": tune.choice([13, 15, 17]),  # Groter voor temporale patronen
        "layer_1_pool": "max",
        
        # Layer 2 - Nog groter (rond 15-19)
        "layer_2_out_channels": tune.choice([96, 128, 160]),
        "layer_2_kernel_size": tune.choice([15, 17, 19]),  # Groot voor complexe patronen
        "layer_2_pool": "avg",
        
        # Layer 3 - Grootste (rond 17-21)
        "layer_3_out_channels": tune.choice([128, 160, 192]),
        "layer_3_kernel_size": tune.choice([17, 19, 21]),  # Grootste voor high-level features
        "layer_3_pool": tune.choice(["none", "avg"]),
        
        # FC layers
        "fc1_size": tune.choice([256, 320, 384]),
        "fc2_size": tune.choice([192, 256, 320]),
        "fc3_size": tune.choice([80, 96, 128]),
        
        # Lagere dropout zoals gevraagd
        "dropout": tune.uniform(0.1, 0.25),  # Veel lager dan voorheen (was 0.35-0.5)
        
        # Architecture features - attention altijd false
        "squeeze_excite": tune.choice([False, True]),
        
        # Training parameters
        "lr": tune.loguniform(8e-4, 4e-3),
        "weight_decay": tune.loguniform(1e-5, 5e-4),
    }

    # Search algorithm
    search_alg = HyperOptSearch(metric="Recall_macro", mode="max")
    
    scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        grace_period=8,
        reduction_factor=2,
        max_t=MAX_EPOCHS,
    )

    reporter = CLIReporter(max_progress_rows=15)
    reporter.add_metric_column("Accuracy")
    reporter.add_metric_column("F1Score_macro")
    reporter.add_metric_column("Recall_macro")
    reporter.add_metric_column("Precision_macro")

    # Run optimization
    analysis = tune.run(
        train,
        config=config,
        metric="Recall_macro",
        mode="max",
        num_samples=NUM_SAMPLES,
        search_alg=search_alg,
        scheduler=scheduler,
        progress_reporter=reporter,
        storage_path=str(Path("logs/ray_improved").resolve()),
        verbose=1,
        max_concurrent_trials=2,
        raise_on_failed_trial=False
    )

    # Results analysis
    best_config = analysis.get_best_config(metric="Recall_macro", mode="max")
    best_result = analysis.best_result
    
    print("\n" + "="*80)
    print("🎯 IMPROVED ECG CNN OPTIMIZATION RESULTS")
    print("="*80)
    print(f"🏆 Best Recall:    {best_result['Recall_macro']:.4f} ({best_result['Recall_macro']*100:.2f}%)")
    print(f"📊 Best Accuracy:  {best_result['Accuracy']:.4f} ({best_result['Accuracy']*100:.2f}%)")
    print(f"⚖️  Best F1 Score:  {best_result['F1Score_macro']:.4f}")
    print(f"🎯 Best Precision: {best_result['Precision_macro']:.4f}")
    
    print("\n" + "="*60)
    print("🏆 OPTIMIZED CONFIGURATION:")
    print("="*60)
    
    # Show the winning architecture met grote kernel sizes
    print(f"\n🏗️  Architecture (grote kernels voor ECG):")
    for i in range(4):
        channels = best_config[f"layer_{i}_out_channels"]
        kernel = best_config[f"layer_{i}_kernel_size"]
        pool = best_config[f"layer_{i}_pool"]
        print(f"   Layer {i}: {channels:3d} channels, kernel {kernel:2d} (groot!), pool '{pool}'")
    
    fc_sizes = [best_config["fc1_size"], best_config["fc2_size"], best_config["fc3_size"], 5]
    print(f"   FC layers: {' → '.join(map(str, fc_sizes))}")
    
    print(f"\n⚙️  Training (aangepast):")
    print(f"   Learning rate:   {best_config['lr']:.4f}")
    print(f"   Weight decay:    {best_config['weight_decay']:.5f}")
    print(f"   Dropout:         {best_config['dropout']:.3f} (lager!)")
    print(f"   LR factor:       0.1 (agressiever)")
    print(f"   Patience:        3 (normaal)")
    
    print(f"\n🔧 Features (vereenvoudigd):")
    print(f"   Squeeze-Excite:  {best_config['squeeze_excite']}")
    print(f"   Attention:       False (uitgeschakeld)")
    print(f"   Skip layers:     Weggehaald (alleen residual blocks)")
    print(f"   Label smoothing: Weggehaald (te complex)")
    
    # Save results
    results = {
        "performance": {
            "recall": float(best_result['Recall_macro']),
            "accuracy": float(best_result['Accuracy']),
            "f1_score": float(best_result['F1Score_macro']),
            "precision": float(best_result['Precision_macro'])
        },
        "winning_configuration": best_config,
        "key_improvements": [
            "Veel grotere kernel sizes (11-21) voor ECG temporale patronen",
            "Lagere dropout (10-25%) voor minder agressieve regularisatie",
            "LR factor 0.1 voor snellere learning rate decay",
            "Attention uitgeschakeld voor eenvoud",
            "Skip layers mechanisme weggehaald",
            "Label smoothing weggehaald voor simplificatie"
        ]
    }
    
    with open("improved_ecg_cnn_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Complete results saved to 'improved_ecg_cnn_results.json'")
    print(f"🎯 Final Score: {best_result['Recall_macro']*100:.2f}% Recall")
    print("✨ Grote kernels + lagere dropout + eenvoudiger architectuur!")

    ray.shutdown()
from pathlib import Path
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import torch
from loguru import logger
import json

from src.models import ModularCNN
from load_heart_data import get_heart_streamers
from src import metrics
from mltrainer import Trainer, TrainerSettings, ReportTypes

NUM_SAMPLES = 20
MAX_EPOCHS = 15

def train(config):
    """Training function based on your working code but optimized"""
    # Generate conv_layers from hyperparameters (exactly like your working version)
    conv_layers = []
    for i in range(config["num_conv_layers"]):
        conv_layers.append({
            "out_channels": config[f"layer_{i}_out_channels"],
            "kernel_size": config[f"layer_{i}_kernel_size"],
            "pool": config.get(f"layer_{i}_pool", "none")
        })

    # Model configuration (exactly like your working version)
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

    # Enhanced loss function for better performance
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))
    
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

    # Training settings based on your working version
    settings = TrainerSettings(
        epochs=MAX_EPOCHS,
        metrics=metric_list,
        logdir=Path("logs/heart1D_optimized"),
        train_steps=len(trainstreamer),
        valid_steps=len(validstreamer),
        reporttypes=[ReportTypes.RAY],
        scheduler_kwargs={"factor": 0.5, "patience": 4},
        optimizer_kwargs={
            "lr": config["lr"], 
            "weight_decay": config["weight_decay"]
        },
        earlystop_kwargs=None,  # Keep simple like your working version
    )

    # Use AdamW for better weight decay handling
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

    # Optimized config based on your successful 96% result
    config = {
        # Data parameters (keep your successful values)
        "data_dir": str(Path("../data").resolve()),
        "dataset_name": "heart_big",
        "target_count": 15000,
        "batch_size": 32,  # Keep your successful batch size
        
        # Model architecture (fine-tune around your best results)
        "input_channels": 1,
        "output": 5,
        "num_conv_layers": 4,
        
        # Layer 0 - optimize around your successful values (64 channels, 7 kernel)
        "layer_0_out_channels": tune.choice([48, 64, 80]),  # Around your 64
        "layer_0_kernel_size": tune.choice([5, 7, 9]),      # Around your 7
        "layer_0_pool": "max",  # Keep your successful "max"
        
        # Layer 1 - optimize around your successful values (64 channels, 7 kernel)
        "layer_1_out_channels": tune.choice([64, 80]),      # Around your 64
        "layer_1_kernel_size": tune.choice([5, 7]),         # Around your 7
        "layer_1_pool": "max",  # Keep your successful "max"
        
        # Layer 2 - optimize around your successful values (96 channels, 3 kernel)
        "layer_2_out_channels": tune.choice([80, 96, 112]), # Around your 96
        "layer_2_kernel_size": tune.choice([3, 5]),         # Around your 3
        "layer_2_pool": "avg",  # Keep your successful "avg"
        
        # Layer 3 - optimize around your successful values (160 channels, 3 kernel)
        "layer_3_out_channels": tune.choice([128, 160, 192]), # Around your 160
        "layer_3_kernel_size": 3,  # Keep your successful 3
        "layer_3_pool": tune.choice(["none", "avg"]),  # Test both
        
        # FC layers - optimize around your successful values
        "fc1_size": tune.choice([256, 320, 384]),    # Around your 256
        "fc2_size": tune.choice([192, 256, 320]),    # Around your 256
        "fc3_size": tune.choice([80, 96, 128]),      # Around your 96
        
        # Regularization - optimize around your successful dropout (0.42)
        "dropout": tune.uniform(0.35, 0.5),  # Around your 0.42
        "label_smoothing": tune.uniform(0.0, 0.08),  # New feature for better generalization
        
        # Architecture features
        "squeeze_excite": tune.choice([False, True]),
        "attention": tune.choice([False, True]),
        "skip_layers": [],  # Keep simple for now
        
        # Training parameters
        "lr": tune.loguniform(8e-4, 4e-3),      # Good range for your problem
        "weight_decay": tune.loguniform(1e-5, 5e-4),  # Regularization
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
        storage_path=str(Path("logs/ray_optimized").resolve()),
        verbose=1,
        max_concurrent_trials=2,
        raise_on_failed_trial=False
    )

    # Enhanced results analysis
    best_config = analysis.get_best_config(metric="Recall_macro", mode="max")
    best_result = analysis.best_result
    
    print("\n" + "="*80)
    print("🎯 PERFORMANCE OPTIMIZATION RESULTS")
    print("="*80)
    print(f"🏆 Best Recall:    {best_result['Recall_macro']:.4f} ({best_result['Recall_macro']*100:.2f}%)")
    print(f"📊 Best Accuracy:  {best_result['Accuracy']:.4f} ({best_result['Accuracy']*100:.2f}%)")
    print(f"⚖️  Best F1 Score:  {best_result['F1Score_macro']:.4f}")
    print(f"🎯 Best Precision: {best_result['Precision_macro']:.4f}")
    
    # Calculate improvement
    baseline_recall = 0.96
    improvement = (best_result['Recall_macro'] - baseline_recall) * 100
    
    if best_result['Recall_macro'] > baseline_recall:
        print(f"\n🚀 SUCCESS! Improvement: +{improvement:.2f} percentage points")
        print(f"   New Performance: {best_result['Recall_macro']*100:.2f}% (was 96.00%)")
    else:
        print(f"\n📈 Close! Difference: {improvement:+.2f} percentage points from 96%")
    
    print("\n" + "="*60)
    print("🏆 OPTIMIZED CONFIGURATION:")
    print("="*60)
    
    # Show the winning architecture
    print(f"\n🏗️  Architecture:")
    for i in range(4):
        channels = best_config[f"layer_{i}_out_channels"]
        kernel = best_config[f"layer_{i}_kernel_size"]
        pool = best_config[f"layer_{i}_pool"]
        print(f"   Layer {i}: {channels:3d} channels, kernel {kernel}, pool '{pool}'")
    
    fc_sizes = [best_config["fc1_size"], best_config["fc2_size"], best_config["fc3_size"], 5]
    print(f"   FC layers: {' → '.join(map(str, fc_sizes))}")
    
    print(f"\n⚙️  Training:")
    print(f"   Learning rate:   {best_config['lr']:.4f}")
    print(f"   Weight decay:    {best_config['weight_decay']:.5f}")
    print(f"   Dropout:         {best_config['dropout']:.3f}")
    print(f"   Label smoothing: {best_config.get('label_smoothing', 0):.3f}")
    
    print(f"\n🔧 Features:")
    print(f"   Squeeze-Excite:  {best_config['squeeze_excite']}")
    print(f"   Attention:       {best_config['attention']}")
    
    # Save comprehensive results
    results = {
        "performance": {
            "recall": float(best_result['Recall_macro']),
            "accuracy": float(best_result['Accuracy']),
            "f1_score": float(best_result['F1Score_macro']),
            "precision": float(best_result['Precision_macro']),
            "improvement_over_baseline": float(improvement),
            "baseline_recall": baseline_recall
        },
        "winning_configuration": best_config,
        "architecture_summary": {
            "conv_layers": [
                {
                    "layer": i,
                    "out_channels": best_config[f"layer_{i}_out_channels"],
                    "kernel_size": best_config[f"layer_{i}_kernel_size"],
                    "pool": best_config[f"layer_{i}_pool"]
                }
                for i in range(4)
            ],
            "fc_layers": fc_sizes,
            "total_parameters": "estimated"
        },
        "key_improvements": [
            "Label smoothing for better generalization",
            "AdamW optimizer for improved weight decay",
            "Fine-tuned architecture around best performing baseline",
            "Optimized learning rate and regularization"
        ]
    }
    
    with open("performance_optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Complete results saved to 'performance_optimization_results.json'")
    
    # Final summary
    if best_result['Recall_macro'] > 0.97:
        print("🎉 EXCELLENT! Achieved >97% recall!")
    elif best_result['Recall_macro'] > baseline_recall:
        print("✅ SUCCESS! Improved beyond baseline!")
    else:
        print("📊 Good run! Consider trying more samples or different ranges.")
    
    print(f"\n🎯 Final Score: {best_result['Recall_macro']*100:.2f}% Recall")

    ray.shutdown()
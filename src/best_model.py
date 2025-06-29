# focused_long_training.py - 5 strategische modellen met lange training
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

NUM_SAMPLES = 5      # Slechts 5 modellen
MAX_EPOCHS = 40      # Veel langer trainen
PATIENCE = 12        # Meer geduld voor convergentie

def train(config):
    """Training function voor lange, grondige training sessies"""
    
    # Generate conv_layers from hyperparameters
    conv_layers = []
    for i in range(4):  # Altijd 4 layers
        conv_layers.append({
            "out_channels": config[f"layer_{i}_out_channels"],
            "kernel_size": config[f"layer_{i}_kernel_size"],
            "pool": config[f"layer_{i}_pool"]
        })

    # Model configuration
    model_config = {
        "input_channels": 1,
        "output": 5,
        "dropout": config["dropout"],
        "squeeze_excite": config["squeeze_excite"],
        "conv_layers": conv_layers,
        "fc1_size": config["fc1_size"],
        "fc2_size": config["fc2_size"],
        "fc3_size": config["fc3_size"]
    }

    trainstreamer, validstreamer = get_heart_streamers(config)
    model = ModularCNN(model_config)

    # Loss function
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
    logger.info(f"Training model: {config['model_name']}")

    # Training settings voor lange sessies
    settings = TrainerSettings(
        epochs=MAX_EPOCHS,
        metrics=metric_list,
        logdir=Path(f"logs/long_training_{config['model_name']}"),
        train_steps=len(trainstreamer),
        valid_steps=len(validstreamer),
        reporttypes=[ReportTypes.RAY],
        scheduler_kwargs={"factor": 0.5, "patience": 6},    # Geduldig LR decay
        optimizer_kwargs={
            "lr": config["lr"], 
            "weight_decay": config["weight_decay"]
        },
        earlystop_kwargs={"patience": PATIENCE},  # Alleen patience
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

    # 5 STRATEGISCHE MODELLEN - elk test een andere hypothese
    
    model_configs = [
        {
            # Model 1: EXACT replica van beste 96.7% model
            "model_name": "exact_replica",
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
            "squeeze_excite": False,
            "label_smoothing": 0.0,
            "lr": 0.001,
            "weight_decay": 1e-4
        },
        
        {
            # Model 2: Replica + Squeeze Excite
            "model_name": "replica_plus_se",
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
            "squeeze_excite": True,    # ENIGE verschil
            "label_smoothing": 0.0,
            "lr": 0.001,
            "weight_decay": 1e-4
        },
        
        {
            # Model 3: Iets groter model (meer capaciteit)
            "model_name": "scaled_up",
            "layer_0_out_channels": 80,   # Iets meer channels
            "layer_0_kernel_size": 7,
            "layer_0_pool": "max",
            "layer_1_out_channels": 80,
            "layer_1_kernel_size": 7,
            "layer_1_pool": "max", 
            "layer_2_out_channels": 128,  # Meer channels
            "layer_2_kernel_size": 3,
            "layer_2_pool": "avg",
            "layer_3_out_channels": 192,  # Meer channels
            "layer_3_kernel_size": 3,
            "layer_3_pool": "none",
            "fc1_size": 320,              # Grotere FC layers
            "fc2_size": 288,
            "fc3_size": 128,              # Meer final capacity
            "dropout": 0.40,              # Iets minder dropout
            "squeeze_excite": True,
            "label_smoothing": 0.02,
            "lr": 0.0008,                 # Iets lagere LR voor stabiliteit
            "weight_decay": 1.2e-4
        },
        
        {
            # Model 4: Alternatieve kernel strategie
            "model_name": "alt_kernels",
            "layer_0_out_channels": 64,
            "layer_0_kernel_size": 5,     # Start kleiner
            "layer_0_pool": "max",
            "layer_1_out_channels": 80,   # Groei eerder
            "layer_1_kernel_size": 7,
            "layer_1_pool": "max",
            "layer_2_out_channels": 96,
            "layer_2_kernel_size": 5,     # Iets groter dan 3
            "layer_2_pool": "avg",
            "layer_3_out_channels": 160,
            "layer_3_kernel_size": 3,
            "layer_3_pool": "avg",        # avg i.p.v. none
            "fc1_size": 288,
            "fc2_size": 256,
            "fc3_size": 96,
            "dropout": 0.41,
            "squeeze_excite": False,
            "label_smoothing": 0.01,
            "lr": 0.0012,
            "weight_decay": 8e-5
        },
        
        {
            # Model 5: Conservative but refined
            "model_name": "conservative_refined", 
            "layer_0_out_channels": 56,   # Iets kleiner start
            "layer_0_kernel_size": 7,
            "layer_0_pool": "max",
            "layer_1_out_channels": 64,
            "layer_1_kernel_size": 7,
            "layer_1_pool": "max",
            "layer_2_out_channels": 88,   # Tussen 80-96
            "layer_2_kernel_size": 3,
            "layer_2_pool": "avg",
            "layer_3_out_channels": 144,  # Tussen 128-160
            "layer_3_kernel_size": 3,
            "layer_3_pool": "none",
            "fc1_size": 240,              # Iets kleiner FC
            "fc2_size": 240,
            "fc3_size": 96,               # Magic number behouden
            "dropout": 0.43,              # Iets meer regularization
            "squeeze_excite": True,       # SE voor extra power
            "label_smoothing": 0.03,
            "lr": 0.0009,
            "weight_decay": 1.5e-4
        }
    ]
    
    # Add shared parameters to each config
    base_config = {
        "data_dir": str(Path("../data").resolve()),
        "dataset_name": "heart_big", 
        "target_count": 15000,
        "batch_size": 32,
        "input_channels": 1,
        "output": 5,
        "num_conv_layers": 4
    }
    
    # Merge configs
    for config in model_configs:
        config.update(base_config)
    
    # Convert to tune format
    config_grid = tune.grid_search(model_configs)

    # Minimal scheduler - laat modellen volledig uittrainen
    scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        grace_period=25,  # Veel geduld voordat pruning
        reduction_factor=2,
        max_t=MAX_EPOCHS,
    )

    reporter = CLIReporter(max_progress_rows=10)
    reporter.add_metric_column("Accuracy")
    reporter.add_metric_column("F1Score_macro") 
    reporter.add_metric_column("Recall_macro")
    reporter.add_metric_column("Precision_macro")

    # Run focused training
    analysis = tune.run(
        train,
        config=config_grid,
        metric="Recall_macro",
        mode="max",
        scheduler=scheduler,
        progress_reporter=reporter,
        storage_path=str(Path("logs/focused_long_training").resolve()),
        verbose=2,
        max_concurrent_trials=3,  # Een tegelijk voor stabiele training
        raise_on_failed_trial=False
    )

    # Comprehensive results analysis
    results = analysis.results_df.sort_values("Recall_macro", ascending=False)
    
    print("\n" + "="*80)
    print("🎯 FOCUSED LONG TRAINING RESULTS (40 epochs)")
    print("="*80)
    
    print(f"\n📊 RANKING (by Recall):")
    print("-" * 60)
    for i, (_, row) in enumerate(results.head().iterrows()):
        config_name = row['config/model_name']
        recall = row['Recall_macro']
        accuracy = row['Accuracy'] 
        f1 = row['F1Score_macro']
        print(f"{i+1}. {config_name:20s} | Recall: {recall:.4f} | Acc: {accuracy:.4f} | F1: {f1:.4f}")
    
    # Best model analysis
    best_result = results.iloc[0]
    best_model = best_result['config/model_name']
    best_recall = best_result['Recall_macro']
    
    print(f"\n🏆 WINNER: {best_model}")
    print(f"🎯 Best Recall: {best_recall:.4f} ({best_recall*100:.2f}%)")
    print(f"📊 Best Accuracy: {best_result['Accuracy']:.4f}")
    print(f"⚖️  Best F1: {best_result['F1Score_macro']:.4f}")
    
    # Performance comparison
    baseline = 0.967
    improvement = (best_recall - baseline) * 100
    
    if best_recall > baseline:
        print(f"\n🚀 SUCCESS! +{improvement:.2f} percentage points improvement!")
    else:
        print(f"\n📈 Close: {improvement:+.2f} percentage points from baseline")
    
    # Model insights
    print(f"\n🔍 INSIGHTS:")
    se_models = results[results['config/squeeze_excite'] == True]
    no_se_models = results[results['config/squeeze_excite'] == False]
    
    if len(se_models) > 0 and len(no_se_models) > 0:
        se_avg = se_models['Recall_macro'].mean()
        no_se_avg = no_se_models['Recall_macro'].mean()
        print(f"   SE blocks: {se_avg:.4f} avg recall")
        print(f"   No SE:     {no_se_avg:.4f} avg recall")
        print(f"   SE Effect: {'+' if se_avg > no_se_avg else ''}{(se_avg - no_se_avg)*100:.2f} percentage points")
    
    # Best configuration details
    best_config_dict = {k.replace('config/', ''): v for k, v in best_result.items() if k.startswith('config/')}
    
    print(f"\n🏗️  WINNING ARCHITECTURE ({best_model}):")
    print("-" * 40)
    for i in range(4):
        ch = best_config_dict[f'layer_{i}_out_channels']
        k = best_config_dict[f'layer_{i}_kernel_size'] 
        p = best_config_dict[f'layer_{i}_pool']
        print(f"   Layer {i}: {ch:3d} channels, kernel {k}, pool '{p}'")
    
    fc1 = best_config_dict['fc1_size']
    fc2 = best_config_dict['fc2_size'] 
    fc3 = best_config_dict['fc3_size']
    print(f"   FC: {fc1} → {fc2} → {fc3} → 5")
    
    print(f"\n⚙️  TRAINING PARAMS:")
    print(f"   Dropout: {best_config_dict['dropout']:.3f}")
    print(f"   SE blocks: {best_config_dict['squeeze_excite']}")
    print(f"   Label smoothing: {best_config_dict['label_smoothing']:.3f}")
    print(f"   Learning rate: {best_config_dict['lr']:.4f}")
    print(f"   Weight decay: {best_config_dict['weight_decay']:.2e}")

    # Save comprehensive results
    final_results = {
        "experiment_type": "focused_long_training",
        "max_epochs": MAX_EPOCHS,
        "num_models": NUM_SAMPLES,
        "winner": {
            "model_name": best_model,
            "recall": float(best_recall),
            "accuracy": float(best_result['Accuracy']),
            "f1_score": float(best_result['F1Score_macro']),
            "configuration": best_config_dict
        },
        "all_results": [
            {
                "model_name": row['config/model_name'],
                "recall": row['Recall_macro'],
                "accuracy": row['Accuracy'],
                "f1_score": row['F1Score_macro']
            }
            for _, row in results.iterrows()
        ],
        "insights": {
            "baseline_recall": baseline,
            "improvement": float(improvement),
            "convergence_epochs": "40 (with early stopping)",
            "key_learnings": [
                f"Best model: {best_model}",
                f"Long training beneficial: {MAX_EPOCHS} epochs vs 15",
                "Patient early stopping allows full convergence",
                f"SE blocks effect: measured across {len(se_models)} vs {len(no_se_models)} models"
            ]
        }
    }
    
    with open("focused_long_training_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n📁 Results saved to 'focused_long_training_results.json'")
    print(f"🎯 Best Score: {best_recall*100:.2f}% Recall")
    print(f"⏱️  Training completed: 5 models × {MAX_EPOCHS} epochs each")
    print("✨ Deep convergence analysis complete!")

    ray.shutdown()
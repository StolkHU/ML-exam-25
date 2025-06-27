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

# KORTE EN STABIELE SETTINGS
NUM_SAMPLES = 25     # Genoeg voor goede variatie
MAX_EPOCHS = 20       # ← VEEL KORTER: was 12, nu 8
STEPS_PER_EPOCH = 3

def train(config):
    trainstreamer, validstreamer = get_heart_streamers(config)
    
    model = ModularCNN(config)

    # Geen class weights - je hebt bewezen dat dit beter werkt
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Minimale metrics voor snelheid
    metric_list = [
        metrics.Accuracy(),
        metrics.Recall(average="macro")
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Snelle training - minder data per epoch
    steps_per_mini_epoch = len(trainstreamer) // STEPS_PER_EPOCH

    settings = TrainerSettings(
        epochs=MAX_EPOCHS,
        metrics=metric_list,
        logdir=Path("logs/stable_fast"),
        train_steps=steps_per_mini_epoch,  
        valid_steps=len(validstreamer) // 3,  # Ook validation sneller
        reporttypes=[ReportTypes.RAY], 
        scheduler_kwargs={"factor": 0.8, "patience": 3},       # ← VROEGER STOPPEN: patience 2
        optimizer_kwargs={
            "lr": config["lr"], 
            "weight_decay": config["weight_decay"],
            "betas": (0.9, 0.999),
            "eps": 1e-8
        },
        earlystop_kwargs={"patience": 3, "verbose": False, "delta": 0.005, "save": True}  # ← VEEL VROEGER
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
    ray.init(
        object_store_memory=1_200_000_000,  # Minder memory voor snelheid
        _memory=3_000_000_000,
        num_cpus=4,
    )

    # STABIELE CONFIG - gebaseerd op je beste resultaten
    config = {
        "data_dir": str(Path("../data").resolve()),
        "dataset_name": "heart_big", 
        "target_count": 15000,  
        # STABIELE BATCH SIZE tegen oscillatie
        "batch_size": tune.choice([32, 36]),  # ← GROTER: was 28 (te klein = meer noise)
        
        "input_channels": 1,
        "output": 5,
        
        # STABIELE REGULARISATIE - rond je 96% waarde
        "dropout": tune.uniform(0.35, 0.45),  # ← ROND je 0.42 van 96% config
        
        # ARCHITECTURE - TERUG naar 96% baseline
        "use_skip": tune.choice([False, True]),     # ← VARIEER: je 96% had False
        "use_attention": False,                     # ← FIXED: uit zoals 96%
        "num_conv_layers": tune.choice([4]),     # ← VARIEER: je 96% had 4
        
        # LAYER CONFIG - ROND je 96% pattern: 64→64→96→160
        
        "layer_0_out_channels": tune.choice([60, 64, 68]),     # Rond 64 van 96%
        "layer_0_kernel_size": tune.choice([7]),              # ← FIXED: 7 zoals 96%
        "layer_0_pool": "max",
        
        "layer_1_out_channels": tune.choice([60, 64, 68]),     # Rond 64 van 96%
        "layer_1_kernel_size": tune.choice([7]),              # ← FIXED: 7 zoals 96%
        "layer_1_pool": "max",
        
        "layer_2_out_channels": tune.choice([88, 96, 104]),    # Rond 96 van 96%
        "layer_2_kernel_size": tune.choice([3]),              # ← FIXED: 3 zoals 96%
        "layer_2_pool": "avg",
        
        "layer_3_out_channels": tune.choice([152, 160, 168]),  # Rond 160 van 96%
        "layer_3_kernel_size": 3,
        "layer_3_pool": "none",
        
        # FC LAYERS - zoals 96% config
        "fc1_size": tune.choice([240, 256, 272]),      # Rond 256 van 96%
        "fc2_size": tune.choice([240, 256, 272]),      # Rond 256 van 96%  
        "fc3_size": tune.choice([88, 96, 104]),        # Rond 96 van 96%
        
        # OPTIMIZER - terug naar 96% baseline range
        "lr": tune.choice([8e-4, 1e-3, 1.2e-3]),              # ← TERUG naar werkende range
        "weight_decay": tune.choice([5e-5, 1e-4, 2e-4]),      # ← Standard range
    }
    
    search_alg = HyperOptSearch(metric="Recall_macro", mode="max")
    
    # ZEER AGRESSIEVE SCHEDULER - stop oscillerende models SNEL
    scheduler = AsyncHyperBandScheduler(
        max_t=MAX_EPOCHS,
        grace_period=6,         # ← ZEER KORT: stop na 3 epochs als slecht
        reduction_factor=3      # ← AGRESSIEF: gooi veel configs weg
    )

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")
    reporter.add_metric_column("Recall_macro")

    analysis = tune.run(
        train,
        config=config,
        metric="Recall_macro",
        mode="max", 
        num_samples=NUM_SAMPLES,
        search_alg=search_alg,
        scheduler=scheduler,
        progress_reporter=reporter,
        storage_path=str(Path("logs/ray_stable_fast").resolve()),
        verbose=1,
        max_concurrent_trials=3,
        resources_per_trial={
            "cpu": 1,
            "gpu": 0
        },
        raise_on_failed_trial=False,
        resume="AUTO"
    )
    
    best_config = analysis.get_best_config(metric="Recall_macro", mode="max")
    print("\nBESTE STABIELE CONFIGURATIE:")
    print("="*50)
    for key, value in best_config.items():
        if not key.startswith('data_'):
            print(f"{key}: {value}")
    
    print(f"\nBeste Recall: {analysis.best_result['Recall_macro']:.4f}")
    
    # Sla alleen de beste op voor finale training
    import json
    with open("best_stable_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    
    print("\n🎯 Voor finale model training:")
    print(f"python train_final.py --config best_stable_config.json")

    ray.shutdown()

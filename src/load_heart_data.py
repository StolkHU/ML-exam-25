from pathlib import Path
from typing import Tuple
import torch
from mltrainer.preprocessors import BasePreprocessor
from mads_datasets.base import BaseDatastreamer
from src.datasets import HeartDataset1D


def get_heart_streamers(config: dict) -> Tuple[BaseDatastreamer, BaseDatastreamer]:
    """
    Laadt de Arrhythmia dataset, past sampling toe en retourneert train en valid streamers.

    Args:
        config (dict): Configuratie met pad en sampling parameters.
            Vereist keys:
                - 'data_dir': pad naar de map met .parq bestanden
                - 'dataset_name': naam van de dataset (bijv. 'heart_big')
                - 'target_count': aantal samples per klasse na sampling
                - 'batch_size': batchgrootte voor de streamers

    Returns:
        Tuple[BaseDatastreamer, BaseDatastreamer]: trainstreamer, validstreamer
    """
    data_dir = Path(config["data_dir"])
    dataset_name = config["dataset_name"]
    target_count = config.get("target_count", 15000)
    batch_size = config.get("batch_size", 32)

    trainfile = data_dir / f"{dataset_name}_train.parq"
    testfile = data_dir / f"{dataset_name}_test.parq"

    traindataset = HeartDataset1D(trainfile, target="target")
    testdataset = HeartDataset1D(testfile, target="target")

    # Sampling per klasse
    labels = torch.unique(traindataset.y)
    all_indices = []

    for label in labels:
        class_indices = (traindataset.y == label).nonzero().squeeze()
        if len(class_indices.shape) == 0:
            class_indices = class_indices.unsqueeze(0)

        current_count = len(class_indices)
        if current_count > target_count:
            selected = class_indices[torch.randperm(current_count)[:target_count]]
        elif current_count < target_count:
            selected = class_indices[torch.randint(0, current_count, (target_count,))]
        else:
            selected = class_indices

        all_indices.append(selected)

    all_indices = torch.cat(all_indices)
    perm = torch.randperm(len(all_indices))
    all_indices = all_indices[perm]

    traindataset.x = traindataset.x[all_indices]
    traindataset.y = traindataset.y[all_indices]

    # Streamers
    trainstreamer = BaseDatastreamer(
        traindataset, preprocessor=BasePreprocessor(), batchsize=batch_size
    )
    validstreamer = BaseDatastreamer(
        testdataset, preprocessor=BasePreprocessor(), batchsize=batch_size
    )

    return trainstreamer, validstreamer

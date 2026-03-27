# build_dataset.py
from dataclasses import dataclass
from typing import Dict, Any
import torch.multiprocessing as mp
import hydra
from omegaconf import DictConfig, OmegaConf
from data_management.dataset_handler import build_dataset
from hydra.utils import to_absolute_path

mp.set_sharing_strategy('file_descriptor')  # avoids POSIX semaphores


@hydra.main(config_path="../conf", config_name="config")
def main(configs: DictConfig):
    print("Active configuration:\n", OmegaConf.to_yaml(configs))
    path_masks = to_absolute_path(configs.dataset.paths.PATH_MASKS)
    path_rgb   = to_absolute_path(configs.dataset.paths.PATH_RGB)
    path_metadata = to_absolute_path(configs.dataset.paths.PATH_METADATA)

    # Build dataset with structured configs
    build_dataset(
        path_masks=path_masks,
        path_rgb=path_rgb,
        path_metadata=path_metadata,
        configs=configs
    )

if __name__ == "__main__":
    main()

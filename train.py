import argparse
import os
from datetime import datetime
from enum import Enum
from pathlib import Path

from typing import Optional

import torch
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


from dataloaders import CIFAR10DataModule
from model import LiNet

logger.remove()
logger.add(
    os.path.join(os.path.dirname(__file__), 'logs', os.path.split(os.path.splitext(__file__)[0])[-1] + "-{time}.log"),
    rotation='2MB', compression="zip", enqueue=True, colorize=False
)

class AutoscalingTechnique(str, Enum):
    power = 'power'
    binsearch = 'binsearch'

def main(
        image_folder: Path,
        save_to_folder: Path,
        gpus: Optional[int] = None,
        epochs: int = 1,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        batch_size: int = 2,
        autoscale_batch_size: Optional[AutoscalingTechnique] = None,
        split_ratio: float = 0.9,
        dropout_rate: float = 0.2,
        num_workers: int = 0):
    """[summary]

    Args:
        image_folder (Path, optional): [description]. Defaults to typer.Option(...).
        save_to_folder (Path, optional): [description]. Defaults to typer.Option(...).
        epochs (int, optional): [description]. Defaults to 5.
        learning_rate (float, optional): [description]. Defaults to 0.001.
        momentum (float, optional): [description]. Defaults to 0.001.
        batch_size (int, optional): [description]. Defaults to 64.

    Returns:
        [type]: [description]
    """
    image_folder.mkdir(parents=True, exist_ok=True)
    save_to_folder.mkdir(parents=True, exist_ok=True)

    model = LiNet(learning_rate=learning_rate, momentum=momentum, batch_size=batch_size, dropout_rate=dropout_rate)
    # Enable Stochastic Weight Averaging using the callback
    # swa = StochasticWeightAveraging(swa_epoch_start=0.75)
    model_checkpoint = ModelCheckpoint(
        monitor='valid/loss',
        save_top_k=1,
        mode='min'
    )
    callbacks=[model_checkpoint]
    trainer = Trainer(max_epochs=epochs,
                      gpus=gpus,
                      logger=WandbLogger(project="dope image classifier", entity="bloodclot-inc"),
                      log_every_n_steps=1,
                      auto_scale_batch_size=autoscale_batch_size,
                      callbacks=callbacks)

    cifar10 = CIFAR10DataModule(data_dir=image_folder.as_posix(),
                                batch_size=batch_size,
                                split_ratio=split_ratio,
                                num_workers=num_workers)
    trainer.tune(model, cifar10)
    trainer.fit(model, cifar10)
    trainer.test(model, cifar10)


    
    # add a batch dimension for onnx conversion -> (num samples, channels, H, W)
    input_sample = torch.randn(cifar10.dims).unsqueeze(0)
    filepath = save_to_folder / f'cifar10-{int(datetime.now().timestamp())}.onnx'
    model.to_onnx(file_path=filepath, input_sample=input_sample, export_params=True)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This app showcases a dope CIFAR10 classifier')
    parser.add_argument('--image-folder', type=Path)
    parser.add_argument('--save-to-folder', type=Path)
    parser.add_argument('--gpus', nargs='?', type=int)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--autoscale-batch-size', type=AutoscalingTechnique, nargs='?', default=None)
    parser.add_argument('--split_ratio', type=float, default=0.9)
    parser.add_argument('--dropout-rate', type=float, default=0.2)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()
    main(**dict(args._get_kwargs()))

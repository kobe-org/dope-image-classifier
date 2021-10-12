import os
from datetime import datetime
from pathlib import Path

import torch
import typer
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


from dataloaders import CIFAR10DataModule
from model import LitNet

logger.remove()
logger.add(
    os.path.join(os.path.dirname(__file__), 'logs', os.path.split(os.path.splitext(__file__)[0])[-1] + "-{time}.log"),
    rotation='2MB', compression="zip", enqueue=True, colorize=False
)

def main(
        image_folder: Path = typer.Option(...),
        save_to_folder: Path = typer.Option(...),
        epochs: int = 1,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        batch_size: int = 256,
        split_ratio: float = 0.9,
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

    model = LitNet(learning_rate=learning_rate, momentum=momentum)
    trainer = Trainer(max_epochs=epochs, logger=WandbLogger(project="dope image classifier", entity="bloodclot-inc"))
    cifar10 = CIFAR10DataModule(data_dir=image_folder, batch_size=batch_size, split_ratio=split_ratio, num_workers=num_workers)
    # cifar10.setup(stage='fit')
    trainer.fit(model, cifar10)

    trainer.test(model, cifar10)
    
    # add a batch dimension for onnx conversion -> (num samples, channels, H, W)
    input_sample = torch.randn(cifar10.dims).unsqueeze(0)
    filepath = save_to_folder / f'cifar10-{int(datetime.now().timestamp())}.onnx'
    model.to_onnx(file_path=filepath, input_sample=input_sample, export_params=True)

    return 0


if __name__ == "__main__":
    typer.run(main)

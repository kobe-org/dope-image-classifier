from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import typer
from fastai.vision.all import *
from PIL import ImageFile
from tqdm import tqdm
from loguru import logger

from model import Net

logger.remove()
logger.add(
    os.path.join(os.path.dirname(__file__), 'logs', os.path.split(os.path.splitext(__file__)[0])[-1] + "-{time}.log"),
    rotation='2MB', compression="zip", enqueue=True, colorize=False
)


def train(model: Net, epochs: int, learning_rate: float, momentum: float, trainloader: DataLoader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    return model


def main(
        image_folder: Path = typer.Option(...),
        save_to_folder: Path = typer.Option(...),
        epochs: int = 5,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        batch_size: int = 64,
        input_name: str = 'image',
        output_name: str = 'label'):
    """[summary]

    Args:
        image_folder (Path, optional): [description]. Defaults to typer.Option(...).
        save_to_folder (Path, optional): [description]. Defaults to typer.Option(...).
        epochs (int, optional): [description]. Defaults to 5.
        learning_rate (float, optional): [description]. Defaults to 0.001.
        momentum (float, optional): [description]. Defaults to 0.001.
        batch_size (int, optional): [description]. Defaults to 64.
        input_name:
        output_name:

    Returns:
        [type]: [description]
    """
    image_folder.mkdir(parents=True, exist_ok=True)
    save_to_folder.mkdir(parents=True, exist_ok=True)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=image_folder.as_posix(), train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=image_folder.as_posix(), train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    model = Net()
    train(model, epochs, learning_rate, momentum, trainloader)

    save_to = save_to_folder / f'cifar10-{int(datetime.now().timestamp())}.pth'
    torch.save(model.state_dict(), save_to.absolute().as_posix())

    return 0


if __name__ == "__main__":
    typer.run(main)

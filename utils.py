from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.onnx
import torchvision
from torch.utils.data import DataLoader


def imshow(img, batch_size: int, classes: tuple, trainloader: DataLoader):
    """

    Args:
        img:
        batch_size:
        classes:
        trainloader:

    Returns:

    """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

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


# Function to Convert to ONNX
def convert_onnx(model, save_to: Path, input_shape: tuple, input_name: str, output_name: str):
    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    # dummy_input = torch.randn(1, 3, 32, 32, requires_grad=True)
    dummy_input = torch.randn(*input_shape, requires_grad=True)

    # Export the model
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      # save_to.as_posix(),  # where to save the model
                      save_to,  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=[input_name],  # the model's input names
                      output_names=[output_name],  # the model's output names
                      dynamic_axes={input_name: {0: 'batch_size'},  # variable length axes
                                    output_name: {0: 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')

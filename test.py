from pathlib import Path
from typing import Tuple, Iterable

import cv2
import numpy as np
import onnx
import onnxruntime as rt
import typer
from PIL import Image
from matplotlib import pyplot as plt


# TODO: Add batch inference

def test(model_checkpoint: Path = typer.Option(...), image_path: Path = typer.Option(...), height: int = 32,
         width: int = 32):
    model = onnx.load(model_checkpoint)
    session = rt.InferenceSession(model.SerializeToString(), None)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    prediction = predict(session, image_path=image_path, size=(height, width), labels=classes)
    return 0


def get_image(path: Path, show: bool = False):
    with Image.open(path) as image:
        image = np.array(image.convert('RGB'))
    if show:
        plt.imshow(image)
        plt.axis('off')
    return image


def preprocess(image: Image, size: Tuple[int, int]):
    image = image / 255.
    image = cv2.resize(image, size)
    # height, width = image.shape[0], image.shape[1]
    # y0 = (height - 224) // 2
    # x0 = (width - 224) // 2
    # image = image[y0: y0 + 224, x0: x0 + 224, :]
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = np.transpose(image, axes=[2, 0, 1])
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image


def predict(session: rt.InferenceSession, image_path: Path, size: Tuple[int, int], labels: Iterable[str]):
    image = get_image(image_path, show=False)
    image = preprocess(image, size=size)
    ort_inputs = {session.get_inputs()[0].name: image}
    preds = session.run(None, ort_inputs)[0]
    preds = np.squeeze(preds)
    a = np.argsort(preds)[::-1]
    return labels[a[0]], preds[a[0]]


if __name__ == '__main__':
    typer.run(test)

from pathlib import Path
import json
import os
import time
import redisai as rai
import ml2rt
from skimage import io
from skimage.transform import resize
import typer
from loguru import logger

logger.remove()
logger.add(
    os.path.join(os.path.dirname(__file__), 'logs', os.path.split(
        os.path.splitext(__file__)[0])[-1] + "-{time}.log"),
    rotation='2MB', compression="zip", enqueue=True, colorize=False
)


def serve_model(gpu: bool = False, host: str = 'localhost', port: int = 6379,
                model_path: Path = typer.Option(...), script_path: Path = typer.Option(...),
                image_path: Path = typer.Option(...)
                ):

    device = 'gpu' if gpu else 'cpu'

    con = rai.Client(host=host, port=port)

    # pt_model_path = '../models/pytorch/imagenet/resnet50.pt'
    # script_path = '../models/pytorch/imagenet/data_processing_script.txt'
    # img_path = '../data/cat.jpg'

    # class_idx = json.load(open("../data/imagenet_classes.json"))
    class_idx = {str(index): class_ for index, class_ in enumerate(['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])}

    image = io.imread(image_path)
    image = resize(image, (32, 32))

    pt_model = ml2rt.load_model(model_path)
    script = ml2rt.load_script(script_path)

    out1 = con.modelset('model', 'onnx', device, pt_model)
    out2 = con.scriptset('script', device, script)
    a = time.time()
    out3 = con.tensorset('image', image)
    out4 = con.scriptrun('script', 'pre_process_3ch', 'image', 'temp1')
    out5 = con.modelrun('model', 'temp1', 'temp2')
    out6 = con.scriptrun('script', 'post_process', 'temp2', 'out')
    final = con.tensorget('out')
    print(final)
    ind = final[0]
    # logger.info(f'{(ind, class_idx[str(ind)])}\n{time.time() - a}')
    print(f'{ind}\n{time.time() - a}')
    # print(time.time() - a)


if __name__ == '__main__':
    typer.run(serve_model)

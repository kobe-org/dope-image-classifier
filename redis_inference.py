from datetime import timedelta
from pathlib import Path
import os
import time
from typing import List
import numpy as np
import redisai as rai
import ml2rt
from skimage.transform import resize
from loguru import logger
from pydantic import BaseModel

logger.remove()
logger.add(
    os.path.join(
        os.path.dirname(__file__),
        "logs",
        os.path.split(os.path.splitext(__file__)[0])[-1] + "-{time}.log",
    ),
    rotation="2MB",
    compression="zip",
    enqueue=True,
    colorize=False,
)


class PredictionScore(BaseModel):
    class_name: str
    score: float

class Prediction(BaseModel):
    detail: List[PredictionScore]
    predicted_class_index: int
    predicted_class: str
    processing_time: timedelta

    class Config:
        arbitrary_types_allowed = True


def serve_model(gpu: bool, host: str, port: int, model_path: Path, script_path: Path):

    device = "gpu" if gpu else "cpu"

    con = rai.Client(host=host, port=port)
    pt_model = ml2rt.load_model(model_path)
    script = ml2rt.load_script(script_path)
    out1 = con.modelset("model", "onnx", device, pt_model)
    out2 = con.scriptset("script", device, script)

    return con


def predict(con, image):
    classes = [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # image = io.imread(image_path)
    image = resize(image, (32, 32))

    start_processing_time = time.time()
    out3 = con.tensorset("image", image)
    out4 = con.scriptrun("script", "pre_process_3ch", "image", "temp1")
    out5 = con.modelrun("model", "temp1", "temp2")
    out6 = con.scriptrun("script", "get_scores", "temp2", "out")
    scores = con.tensorget("out").tolist()
    
    detailed_scores = [PredictionScore(class_name=key, score=value) for key, value in zip(classes, scores)]
    predicted_class_index = np.argmax(scores)
    predicted_class = classes[predicted_class_index]
    processing_time = timedelta(microseconds=(time.time() - start_processing_time) * 1e6)

    prediction = Prediction(
        detail=detailed_scores,
        predicted_class_index=predicted_class_index,
        predicted_class=predicted_class,
        processing_time=processing_time
    )
    return prediction

from pathlib import Path
import json
import os
import time
import numpy as np
import redisai as rai
import ml2rt
from skimage import io
from skimage.transform import resize
import typer
from loguru import logger
import streamlit as st
from PIL import Image

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


def serve_model(
    gpu: bool,
    host: str,
    port: int,
    model_path: Path,
    script_path: Path
):

    device = "gpu" if gpu else "cpu"

    con = rai.Client(host=host, port=port)
    pt_model = ml2rt.load_model(model_path)
    script = ml2rt.load_script(script_path)
    out1 = con.modelset("model", "onnx", device, pt_model)
    out2 = con.scriptset("script", device, script)

    return con


def predict(con, image):
    class_idx = {
        str(index): class_
        for index, class_ in enumerate(
            [
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
        )
    }

    # image = io.imread(image_path)
    image = resize(image, (32, 32))

    a = time.time()
    out3 = con.tensorset("image", image)
    out4 = con.scriptrun("script", "pre_process_3ch", "image", "temp1")
    out5 = con.modelrun("model", "temp1", "temp2")
    out6 = con.scriptrun("script", "get_score", "temp2", "out")
    final = con.tensorget("out")
    print(final)
    ind = final[0]
    # logger.info(f'{(ind, class_idx[str(ind)])}\n{time.time() - a}')
    print(f"{ind}\n{time.time() - a}")
    return ind


# def main(
#     gpu: bool = False,
#     host: str = "localhost",
#     port: int = 6379,
#     model_path: Path = typer.Option(...),
#     script_path: Path = typer.Option(...),
#     image_path: Path = typer.Option(...),
# ):
model_path = Path(os.path.join(os.path.dirname(__file__), "saved", "cifar10-1632426725.onnx"))
script_path = Path(os.path.join(os.path.dirname(__file__), "data_processing_script.txt"))
con = serve_model(False, "localhost", 6379, model_path, script_path)

st.title("Upload + Classification Example")

# file is uploaded as a bytes object
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # print(uploaded_file.)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict(con, np.array(image))
    st.write(f'{label}')



# if __name__ == "__main__":
#     typer.run(main)
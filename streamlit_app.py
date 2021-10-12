import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from redis_inference import predict, serve_model


# Streamlit encourages well-structured code, like starting execution in a main() function.
def main(gpu: bool, host: str, port: int, model: Path, script: Path):
    con = serve_model(gpu=gpu, host=host, port=port, model_path=model, script_path=script)
    st.title("Dope Image Classifier")
    run_the_app(con)


# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app(con):
    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # In this common pattern, we download data from an endpoint only once.

    # file is uploaded as a bytes object
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.write("")
    
        submit = st.button('Classify')  
        if submit:
            with st.spinner(text="This may take a moment..."):
                prediction = predict(con, np.array(image))
        
                df_detail = pd.DataFrame([{"Class Name": element.class_name, "Score": element.score} for element in  prediction.detail])
                st.dataframe(df_detail.style.highlight_max(axis=0))
                st.write(f'Processing Time: {prediction.processing_time}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This app showcases a dope CIFAR10 classifier')

    parser.add_argument('--model', type=Path)
    parser.add_argument('--script', type=Path)
    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=6379)
    args = parser.parse_args()
    main(args.gpu, args.host, args.port, args.model, args.script)

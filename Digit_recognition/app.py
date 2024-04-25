import base64
import json
import os
import re
import time
import uuid
from io import BytesIO
from pathlib import Path

import tensorflow as tf
import numpy as np
import pandas as pd
import streamlit as st
import PIL
from streamlit_drawable_canvas import st_canvas
from svgpathtools import parse_path
import cv2

def load_model():
    loaded_model = tf.keras.models.load_model('Digit_Recognition_model.keras')
    return loaded_model

model = load_model()

def converter(arr):
    # print(arr.shape)
    new_arr = np.zeros((560, 560))
    temp_arr = np.array([255, 255, 255, 255])
    for i in range(560):
        for j in range(560):
            boole = arr[i][j] == temp_arr
            if(boole.all()):
                new_arr[i][j] = 1
    return new_arr
            
# Specify canvas parameters in application
def full_app():
    st.sidebar.header("Configuration")
    st.markdown(
        """
    Draw on the canvas, get the drawings back to Streamlit!
    * Configure canvas in the sidebar
    * In transform mode, double-click an object to remove it
    * In polygon mode, left-click to add a point, right-click to close the polygon, double-click to remove the latest point
    """
    )

    with st.echo("below"):
        # Specify canvas parameters in application
        drawing_mode = st.sidebar.selectbox(
            "Drawing tool:",
            ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
        )
        stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 20)
        if drawing_mode == "point":
            point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
        stroke_color = st.sidebar.color_picker("Stroke color hex: ", value="#fff")
        bg_color = st.sidebar.color_picker("Background color hex: ", "#000")
        bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
        realtime_update = st.sidebar.checkbox("Update in realtime", True)

        #Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=PIL.Image.open(bg_image) if bg_image else None,
            update_streamlit=realtime_update,
            height = 560,
            width = 560,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == "point" else 0,
            display_toolbar=st.sidebar.checkbox("Display toolbar", True),
            # key="full_app",
        )

        # Do something interesting with the image data and paths
        if canvas_result.image_data is not None:
            old_arr = np.array(canvas_result.image_data)
            new_arr = converter(old_arr)
            img_28x28 = np.array(cv2.resize(new_arr, dsize = (28,28), interpolation= cv2.INTER_LANCZOS4))
            img_array = (img_28x28.flatten())

            img_array  = img_array.reshape(1,-1)
            Digits = ['0','1','2','3','4','5','6','7','8','9']
            st.markdown("#### Click to submit your Canvas")
            click = st.button("Submit")
            if(click):
                prediction = Digits[np.argmax(model.predict([img_array]))]
                st.markdown(f"You typed {prediction}")
            
            
        if canvas_result.json_data is not None:
            objects = pd.json_normalize(canvas_result.json_data["objects"])
            for col in objects.select_dtypes(include=["object"]).columns:
                objects[col] = objects[col].astype("str")
            st.dataframe(objects)
            

full_app()
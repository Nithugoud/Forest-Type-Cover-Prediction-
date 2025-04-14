import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image

rfc = pickle.load(open('rfc.pkl', 'rb'))

# Creating web app
st.title('Forest Cover Type Prediction')
image = Image.open('img.png')
st.image(image, caption='myimage', use_column_width=True)
user_input = st.text_input('Enter all cover type features')

if user_input:
    user_input = user_input.split(',')
    features = np.array([user_input], dtype=np.float64)
    prediction = rfc.predict(features).reshape(1, -1)
    prediction = int(prediction[0])

    # Create the cover type dictionary
    cover_type_dict = {
        1: {"name": "Spruce/Fir", "image": "img_1.jpg"},
        2: {"name": "Lodgepole Pine", "image": "img_2.jpeg"},
        3: {"name": "Ponderosa Pine", "image": "img_3.jpg"},
        4: {"name": "Cottonwood/Willow", "image": "img_4.jpg"},
        5: {"name": "Aspen", "image": "img_5.jpg"},
        6: {"name": "Douglas-fir", "image": "img_6.jpg"},
        7: {"name": "Krummholz", "image": "img_7.jpg"}
    }

    cover_type_info = cover_type_dict.get(prediction)

    if cover_type_info is not None:
        forest_name = cover_type_info["name"]
        forest_image = cover_type_info["image"]

        # Display the cover type card
        col1, col2 = st.columns([2,3])

        with col1:
            st.write("Predicted Cover Type:")
            st.write(f"<h1 style='font-size: 40px; font-weight: bold;'>{forest_name}</h1>", unsafe_allow_html=True)

        with col2:
            final_image = Image.open(forest_image)
            st.image(final_image, caption=forest_name, use_column_width=True)

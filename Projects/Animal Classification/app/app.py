import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# triande model in model folder and save as animal_classification_model.h5
model = tf.keras.models.load_model('animal_classification_model.h5')

class_names = ['Bear','Bird','Cat','Cow','Deer','Dog','Dolphin','Elephant','Giraffe','Horse','Kangaroo','Lion','Panda','Tiger','Zebra']

st.title("🐾 Animal Classification App")

uploaded_file = st.file_uploader("Upload an Animal Image", type=["jpg","png","jpeg"])




if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img = img.resize((128,128))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    
    st.success(f"Predicted Animal: {predicted_class}")
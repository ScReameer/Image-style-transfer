import os
import sys
sys.path.append(".")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import streamlit as st
import tensorflow as tf
import keras
import wget

st.title('Style transfer using AdaIN')

@st.cache_resource
def get_model():
    folder_name='model/'
    model_filename = folder_name + 'model.keras'
    # Download model if not already downloaded
    if not os.path.exists(folder_name): 
        os.mkdir(folder_name)
        url_model = r'https://drive.usercontent.google.com/download?id=1N0t6uhtO4W9tlLvTGLqm_y5beZQmu7sk&export=download&confirm=yes'
        wget.download(url_model, out=model_filename)
    # Load and return pretrained model
    model = keras.models.load_model(model_filename, compile=False)
    model.trainable = False
    return model


def get_prediction(model, content, style, alpha, save_content_colors):
    content_img = tf.expand_dims(tf.image.decode_image(content.getvalue()), 0)
    style_img = tf.expand_dims(tf.image.decode_image(style.getvalue()), 0)
    outputs = model.predict(content_img, style_img, alpha, save_content_colors)[0]
    return outputs.numpy() / 255

model = get_model()
content = st.file_uploader('content_file', key='content_file', type=['png', 'jpg', 'jpeg'])
style = st.file_uploader('style_file', key='style_file', type=['png', 'jpg', 'jpeg'])
alpha = st.slider('Alpha', 0.0, 1.0, 1.0)
save_content_colors = st.checkbox('Save content colors')

if content and style:
    content_column, output_column, style_column = st.columns(3)
    output = get_prediction(model, content, style, alpha, save_content_colors)
    with content_column:
        st.markdown('### Content')
        st.image(content)
        
    with output_column:
        st.markdown('### Generated')
        st.image(output)
        
    with style_column:
        st.markdown('### Style')
        st.image(style)
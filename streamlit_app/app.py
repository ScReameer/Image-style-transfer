import os
import sys
sys.path.append(".")
sys.path.append("..")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import streamlit as st
import tensorflow as tf
import keras
import wget

st.markdown('# <center>Style transfer using AdaIN', unsafe_allow_html=True)

@st.cache_resource
def get_model():
    folder_name='model/'
    model_filename = folder_name + 'model.keras'
    # Download model if not already downloaded
    if not os.path.exists(folder_name): 
        os.mkdir(folder_name)
        url_model = r'https://drive.usercontent.google.com/download?id=1N0t6uhtO4W9tlLvTGLqm_y5beZQmu7sk&export=download&confirm=yes'
        wget.download(url_model, out=model_filename)
    # Load pretrained model
    model = keras.models.load_model(model_filename, compile=False)
    model.trainable = False
    return model

def get_prediction(model, content, style, alpha, save_content_colors):
    # Load images from bytes to tensors, add batch dimension
    content_img = tf.image.decode_image(content.getvalue(), channels=3)[tf.newaxis, ...]
    style_img = tf.image.decode_image(style.getvalue(), channels=3)[tf.newaxis, ...]
    outputs = model.predict(content_img, style_img, alpha, save_content_colors)[0]
    return outputs.numpy() / 255

model = get_model()
content_upload_col, style_upload_col = st.columns(2, gap='large')
allowed_types = ['png', 'jpg', 'jpeg']
# Content upload button
with content_upload_col:
    content = st.file_uploader('**Изображение контента**', key='content_file', type=allowed_types)
# Style upload button
with style_upload_col:
    style = st.file_uploader('**Изображение стиля**', key='style_file', type=allowed_types)
# 'Weight' of style
alpha = st.slider('Alpha', 0.0, 1.0, 1.0)
# Match histogram from content to style before computing AdaIN
save_content_colors = st.checkbox('Save content colors')
# Show results
if content and style:
    with st.spinner():
        output = get_prediction(model, content, style, alpha, save_content_colors)
    st.markdown('### <center>Generated', unsafe_allow_html=True)
    st.image(output)
    content_col, style_col = st.columns(2, gap='large')
    with content_col:
        st.markdown('### <center>Content', unsafe_allow_html=True)
        st.image(content)
    with style_col:
        st.markdown('### <center>Style', unsafe_allow_html=True)
        st.image(style)

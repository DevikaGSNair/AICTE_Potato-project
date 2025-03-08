import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)
st.sidebar.title("plant disease system for sustainable agriculture")
app_mode = st.sidebar.selectbox('select page',['home','disease recognition'])

from PIL import Image
img = Image.open('Disease.png')
st.image(img)

if(app_mode == 'HOME'):
    st.markdown("<h1 style = 'text-align: center;'>Plant disease detection system for sustainable agriculture", unsafe_allow_html=True )

elif(app_mode == 'Disease recognition'):
    st.header('plant disease system for sustainable agriculture')


test_image = st.file_uploader('choose an image:')
if(st.button('Show Image')):
    st.image(test_image,width=4,use_column_width=True)

if(st.button('predict')):
    st.snow()
    st.write('our prediction')
    result_index = model_prediction(test_image)
    class_name = ['Potato___Early_blight','Potato___Late_blight','Potato___healthy']
    st.success('model is predicting its a {}'.format(class_name[result_index]))

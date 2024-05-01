import streamlit as st
import numpy as np
import requests
import json
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import shap
import matplotlib.pyplot as plt

# Normalize image for plotting
def normalize_image(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def get_preprocessing_function(model_name):
    if model_name == 'ResNet50':
        return preprocess_input_resnet
    elif model_name == 'VGG16':
        return preprocess_input_vgg
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def load_model(model_name):
    if model_name == 'ResNet50':
        return ResNet50(weights="imagenet")
    elif model_name == 'VGG16':
        return VGG16(weights="imagenet")
    else:
        return None

# Streamlit app setup
st.title("Image Classification with SHAP Explanation")
st.balloons()

# Model selection
model_option = st.selectbox('Select a model:', ['ResNet50', 'VGG16'])
model = load_model(model_option)
preprocess_input = get_preprocessing_function(model_option)

def f(X):
    tmp = preprocess_input(X.copy())
    return model.predict(tmp)

# Load ImageNet class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
response = requests.get(url)
class_names = [v[1] for v in json.loads(response.text).values()]

# Image uploader
uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
if uploaded_files:
    images = np.array([img_to_array(load_img(file, target_size=(224, 224))) for file in uploaded_files])
    images = preprocess_input(images.copy())

    if images.size > 0:
        images_normalized = np.array([normalize_image(img) for img in images])

        # Change masker to a blur technique
        masker = shap.maskers.Image("blur(10, 10)", images_normalized[0].shape)

        # Tgis Explainer with potentially increased max_evals for better accuracy (accurate prediction)
        explainer = shap.Explainer(f, masker, output_names=class_names)
        shap_values = explainer(images, max_evals=1000, batch_size=50, outputs=shap.Explanation.argsort.flip[:4])
        
        # Visualization and save
        fig, ax = plt.subplots()
        shap.image_plot(shap_values)
        plt.savefig('shap_output.png')
        plt.close(fig)
        
        st.image('shap_output.png', caption='SHAP Output')
else:
    st.text("Please upload one or more images.")

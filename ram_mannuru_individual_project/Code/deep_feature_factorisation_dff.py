import streamlit as st
from PIL import Image
import numpy as np
import cv2
import requests
import numpy as np

import torch
from pytorch_grad_cam import DeepFeatureFactorization
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image, deprocess_image
from torchvision.models import resnet50
from pytorch_grad_cam.utils.image import show_factorization_on_image

def create_labels(concept_scores, top_k=2):
    """ Create a list with the ImageNet category names of the top scoring categories"""
    imagenet_categories_url = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
    labels = eval(requests.get(imagenet_categories_url).text)
    concept_categories = np.argsort(concept_scores, axis=1)[:, ::-1][:, :top_k]
    concept_labels_topk = []
    for concept_index in range(concept_categories.shape[0]):
        categories = concept_categories[concept_index, :]    
        concept_labels = []
        for category in categories:
            score = concept_scores[concept_index, category]
            label = f"{labels[category].split(',')[0]}:{score:.2f}"
            concept_labels.append(label)
        concept_labels_topk.append("\n".join(concept_labels))
    return concept_labels_topk

@st.cache_data
def load_model():
    model = resnet50(pretrained=True)
    model.eval()
    return model

model = load_model()

def visualize_image(model, image_file, n_components=5, top_k=2):
    img = np.array(Image.open(image_file))
    rgb_img_float = np.float32(img) / 255
    input_tensor = preprocess_image(rgb_img_float,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    classifier = model.fc
    dff = DeepFeatureFactorization(model=model, target_layer=model.layer4, 
                                   computation_on_concepts=classifier)
    concepts, batch_explanations, concept_outputs = dff(input_tensor, n_components)
    
    concept_outputs = torch.softmax(torch.from_numpy(concept_outputs), axis=-1).numpy()    
    concept_label_strings = create_labels(concept_outputs, top_k=top_k)
    visualization = show_factorization_on_image(rgb_img_float, 
                                                batch_explanations[0],
                                                image_weight=0.3,
                                                concept_labels=concept_label_strings)
    
    result = np.hstack((img, visualization))
    
    # Just for the Jupyter notebook, so the large images won't weigh a lot:
    if result.shape[0] > 500:
        result = cv2.resize(result, (result.shape[1]//4, result.shape[0]//4))
    
    return result

st.title("Image Visualization with Deep Feature Factorization")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def ppredict(file):
    import numpy as np
    from tensorflow.keras.utils import load_img
    from tensorflow.keras.utils import img_to_array
    from keras.applications.resnet50 import preprocess_input
    from keras.applications.resnet50 import ResNet50
    from keras.applications.imagenet_utils import decode_predictions
    import matplotlib.pyplot as plt

    img = load_img(file, target_size = (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img)
    model = ResNet50(weights="imagenet")
    preds = model.predict(img)
    return decode_predictions(preds, top=2)[0]

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    img = np.array(Image.open(uploaded_file))
    preds = ppredict(uploaded_file)
    st.write("Predicted class using Resnet", preds)
    result_image = visualize_image(model, uploaded_file, n_components=5, top_k=2)
    st.image(result_image, caption="Visualization Result", use_column_width=True)

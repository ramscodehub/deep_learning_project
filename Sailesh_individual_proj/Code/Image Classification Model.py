import streamlit as st
import streamlit.components.v1 as components
from io import StringIO
#import ast
import re
#from lime_explainer import explainer, tokenizer, METHODS
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision import models
from torchvision.transforms import v2
from tqdm import tqdm
from torchvision import models
import torchvision
import tensorflow as tf
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as c_map
from IPython.display import Image, display
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import lime
from lime import lime_image
from lime import submodular_pick
from skimage.segmentation import mark_boundaries
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from lime import lime_image
import shap
import numpy as np
import base64
from io import BytesIO
#torch.random.seed(123)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def predict(image_tensor,selected_framework,model):
    if selected_framework == "PyTorch":
        with torch.no_grad():
            output = model(image_tensor)
        return output
    else:
        return model.predict(image_tensor)
def preprocess_image(image, preprocess_fn, image_size, PyTorch=True,for_model = True):
    if for_model:
        if PyTorch:
            preprocess = eval(preprocess_fn)
            return preprocess(image).unsqueeze(0)
        else:
            image = tf.image.resize(image, [image_size, image_size])
            img = tf.keras.preprocessing.image.img_to_array(image)
            image = img / 255.0
            return np.expand_dims(image,axis=0)
    else:
            if PyTorch:
                preprocess = eval(preprocess_fn)
                return preprocess(image)
            else:
                resized_image = tf.image.resize(image, [image_size, image_size])
                image_tensor = tf.cast(resized_image, dtype=tf.float32)
                image = image_tensor / 255.0
                return image

# App Building
def main():
    st.balloons()
    title_text = 'AI Explainability Dashboard'
    st.markdown(f"<h2 style='text-align: center;'><b>{title_text}</b></h2>", unsafe_allow_html=True)
    st.subheader("LIME for Image Classification Model")
    st.text("")
    # Upload custom PyTorch model (.pt)
    selected_framework = st.radio("Select the Deep Learning framework:", ["PyTorch", "TensorFlow"])
    selected_model = st.radio("Pretrained", [ "Pre-trained"])

    if selected_model == "Pre-trained" and selected_framework in ['TensorFlow','PyTorch']:
        model_architecture_code = st.text_area("Instantiate pre-trained model with corresponding weights. Note: write full library. TensorFlow as tf and torch as torch.")
        st.code(model_architecture_code, language="python")
    else:
        model_architecture_code = True

    image_size = int(st.text_input("Enter the image size for your model (Note: For pre-trained models, it must match with image size that was used to train the model)", value="224"))
    Mean_list = (st.text_input("Enter your desired image normalization - Mean", value="0.5, 0.5, 0.5"))
    Std_list = (st.text_input("Enter your desired image normalization - Standard Deviation", value="0.5, 0.5, 0.5"))
    Mean_list = Mean_list.split(",")
    Std_list = Std_list.split(",")

    preprocess_fn_code = f"torchvision.transforms.Compose([torchvision.transforms.ToTensor(),\ntorchvision.transforms.Resize(({image_size}, {image_size})),\ntorchvision.transforms.Normalize(\nmean={[float(a) for a in Mean_list]},\nstd={[float(a) for a in Std_list]})])"
    st.text("Applied pre-processing")
    st.code(preprocess_fn_code, language="python")
    if model_architecture_code is not None and selected_model == "Custom" and selected_framework == 'PyTorch':
        clean_string = re.sub(r'#.*', '', model_architecture_code)
        clean_string = re.sub(r'(\'\'\'(.|\n)*?\'\'\'|"""(.|\n)*?""")', '', clean_string, flags=re.DOTALL)
        pattern = re.compile(r'class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(.*\):',re.IGNORECASE)
        class_name = pattern.search(clean_string)
        if class_name:
            class_name = class_name.group(1)
    elif model_architecture_code is not None and selected_model in ["Pre-trained + Custom", 'Pre-trained']:
        model_name = st.text_input("The name of the model variable you've assigned (e.g model)", value='model')

    image_file = st.file_uploader("Upload the image you want to explain", type=["jpg", "jpeg", "png"])

    if model_architecture_code and image_file:
        if selected_framework == "PyTorch":
            if not isinstance(model_architecture_code, (bool)):
                exec(model_architecture_code, globals())
            # Load the PyTorch model
            if selected_model == "Pre-trained + Custom":
                # exec(model_architecture_code, globals())
                model = globals()[model_name]
                file = torch.load("model.pt", map_location=torch.device(device))
                model.load_state_dict(file)
            elif selected_model == "Custom":
                # exec(model_architecture_code, globals())
                model = globals()[model_name]
                file = torch.load("model.pt", map_location=torch.device(device))
                model.load_state_dict(file)
            else:
                model = globals()[model_name]
            model.eval()

        elif selected_framework == "TensorFlow":
            if not isinstance(model_architecture_code, (bool)):
                exec(model_architecture_code, globals())
            if selected_model == "Pre-trained":
                model = globals()[model_name]
            else:
                model = tf.keras.models.load_model("model.h5")

        else:
            st.error("Invalid framework selected.")

        # Load and display the image
        image = Image.open(image_file)
        image = image.resize((int(image_size), int(image_size)))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        my_bool = True if selected_framework == "PyTorch" else False
        input_image = preprocess_image(image,preprocess_fn_code,image_size,PyTorch=my_bool,for_model = True)

        # Define a function for model prediction
        pred_orig = predict(input_image,selected_framework,model)
        st.write("Your Predicted Output from the model is as follows:", np.array(pred_orig))


        if st.button("Explain Model"):
            with st.spinner('Calculating LIME Analysis...'):
                if selected_framework == "PyTorch":
                    def batch_predict(images):
                        model.eval()
                        batch = torch.stack(tuple(preprocess_image(i,preprocess_fn_code,image_size=image_size,PyTorch = my_bool,for_model=False) for i in images), dim=0)
                        model.to(device)
                        batch = batch.to(device)
                        logits = model(batch)
                        probs = torch.nn.functional.softmax(logits, dim=1)
                        return probs.detach().cpu().numpy()
                    explainer = lime_image.LimeImageExplainer()
                    exp = explainer.explain_instance(np.array(image),
                                                 batch_predict,
                                                 top_labels=5,
                                                 hide_color=0,
                                                 num_samples=1000)
                else:
                    explainer = lime_image.LimeImageExplainer()
                    exp = explainer.explain_instance(np.array(input_image[0].astype('double')),
                                                     model.predict,
                                                     top_labels=5,
                                                     hide_color=0,
                                                     num_samples=1000)
                temp_1, mask_1 = exp.get_image_and_mask(exp.top_labels[0], positive_only=True,
                                                                    num_features=5, hide_rest=True)
                temp_2, mask_2 = exp.get_image_and_mask(exp.top_labels[0], positive_only=False,
                                                                    num_features=5, hide_rest=False)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
                ax1.imshow(mark_boundaries(temp_1, mask_1))
                ax2.imshow(mark_boundaries(temp_2, mask_2))
                ax1.axis('off')
                ax2.axis('off')
                plt.savefig('mask.png')
                Lime_img1 = Image.open('mask.png')
                st.image(Lime_img1)
                st.write("Image on the left denotes the super-pixels or region-of-interest based on LIME analysis. Classification is done due to the highlighted super-pixels. Image on the right imposes this region-of-interest on original image giving a more intuitive understanding.")
                dict_heatmap = dict(exp.local_exp[exp.top_labels[0]])
                heatmap = np.vectorize(dict_heatmap.get)(exp.segments)
                plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
                plt.colorbar()
                plt.savefig('heatmap.png')
                Lime_img2 = Image.open('heatmap.png')
                st.image(Lime_img2)
                st.write("This section shows a heat-map that displays how important each super-pixel is to get some more granular explaianbility. The legend includes what color-coded regions of interest move the decision of the model. Blue indicates the regions that influences the decision of the model in the predicted class and red indicates the regions that influence the decision to other classes.")

if __name__ == "__main__":
    main()



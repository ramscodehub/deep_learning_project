import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import Model
import tensorflow as tf
import imutils
import matplotlib.pyplot as plt
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input



# Define GradCAM class
class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output.shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        gradModel = Model(inputs=self.model.inputs, outputs=[self.model.get_layer(self.layerName).output, self.model.output])
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]
        grads = tape.gradient(loss, convOutputs)
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        return output

# Define OcclusionSensitivity class
class OcclusionSensitivity:
    def __init__(self, model, classIdx, patch_size=(5, 5), patch_stride=(4, 4), occlusion_value=255):
        self.model = model
        self.classIdx = classIdx
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.occlusion_value = occlusion_value

    def explain(self, image):
        (h, w) = image.shape[:2]
        (patchH, patchW) = self.patch_size
        (strideH, strideW) = self.patch_stride

        heatmap = np.zeros((h, w), dtype="float32")

        for y in range(0, h - patchH + 1, strideH):
            for x in range(0, w - patchW + 1, strideW):
                clone = image.copy()
                clone[y:y + patchH, x:x + patchW] = self.occlusion_value

                preds = self.model.predict(np.expand_dims(clone, axis=0))
                #print("Predictions:", preds) 
                i = self.classIdx
                prob = preds[0][i]

                heatmap[y:y + patchH, x:x + patchW] = prob

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255
        return heatmap.astype("uint8")

def main():
    st.title("XAI with GradCAM and Occlusion Sensitivity")

    # Load pre-trained model
    model = ResNet50(weights="imagenet")

    # Upload image
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Read and preprocess image
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image, (224, 224))
        image = img_to_array(resized)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # Make predictions
        preds = model.predict(image)
        i = np.argmax(preds[0])
        predicted_class = decode_predictions(preds, top=1)[0][0]  # Get top prediction
        predicted_label = predicted_class[1]
        predicted_probability = predicted_class[2]

        # Display classification result
        st.write("Predicted Class:", predicted_label)
        st.write("Probability:", predicted_probability)

        # Compute GradCAM
      
        gradcam = GradCAM(model, i)
        gradcam_heatmap = gradcam.compute_heatmap(image)
        gradcam_heatmap = cv2.resize(gradcam_heatmap, (image.shape[2], image.shape[1]))
        gradcam_heatmap = cv2.applyColorMap(gradcam_heatmap, cv2.COLORMAP_JET)
        # Compute Occlusion Sensitivity
        occlusion = OcclusionSensitivity(model, i)
        occlusion_heatmap = occlusion.explain(resized)

        # Overlay heatmaps on original image
        gradcam_overlay = gradcam.overlay_heatmap(gradcam_heatmap, resized, alpha=0.3)
        occlusion_overlay = cv2.applyColorMap(occlusion_heatmap, cv2.COLORMAP_JET)


        # Display results
        st.image([resized, gradcam_heatmap, gradcam_overlay, occlusion_overlay],
                 caption=["Original Image", "GradCAM Heatmap", "GradCAM Overlayed Image", "Occlusion Sensitivity Heatmap"],
                 width=200)

if __name__ == "__main__":
    main()
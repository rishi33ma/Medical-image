import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from keras.models import load_model, Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import img_to_array

# Load brain tumor classification model
def load_brain_tumor_model():
    saved_model_path = "D:\\Siddhant\\VIT\\T.Y. Sem 2\\DL CP\\brain_tumor_classifier.h5"  
    model = load_model(saved_model_path)
    class_dict = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}
    return model, class_dict

# Load chest X-ray classification model
def load_chest_xray_model():
    model_path = "D:\\Siddhant\\VIT\\T.Y. Sem 2\\DL CP\\chest_model.h5"
    model = load_model(model_path)
    labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
              'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']
    return model, labels

# Define custom loss function for chest X-ray model
def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    def weighted_loss(y_true, y_pred):
        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(y_pred, 'float32')
        loss_pos = -tf.reduce_mean(pos_weights * y_true * tf.math.log(y_pred + epsilon), axis=0)
        loss_neg = -tf.reduce_mean(neg_weights * (1 - y_true) * tf.math.log(1 - y_pred + epsilon), axis=0)
        loss = loss_pos + loss_neg
        return loss
    return weighted_loss

# Register custom loss function for chest X-ray model
tf.keras.utils.get_custom_objects()['weighted_loss'] = get_weighted_loss

# Function to preprocess the image for chest X-ray model
def preprocess_chest_xray_image(image):
    img_pil = Image.open(image)
    img_array = np.array(img_pil)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_array = cv2.resize(img_array, (320, 320))
    img_array = img_array / 255.0
    return img_array

# Function to preprocess the image for brain tumor model
def preprocess_brain_tumor_image(image):
    img = Image.open(image)
    img_resized = img.resize((299, 299))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    if img_array.shape[-1] != 3:
        img_array = np.repeat(img_array, 3, axis=-1)
    return img_array

# Function to make predictions with brain tumor model
def predict_with_brain_tumor_model(image_path, model):
    class_dict = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}
    img_array = preprocess_brain_tumor_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = class_dict[np.argmax(predictions)]
    return predicted_class

# Function to make predictions with chest X-ray model
def predict_with_chest_xray_model(image_path, model, labels):
    img_array = preprocess_chest_xray_image(image_path)
    predictions = model.predict(np.expand_dims(img_array, axis=0))
    top_preds_indices = np.argsort(predictions[0])[::-1][:3]
    top_preds_labels = [labels[i] for i in top_preds_indices]
    return top_preds_labels

def main():
    st.title("Medical Image Analysis")

    # Choose model
    model_type = st.radio("Select Model", ("Brain X-Ray", "Chest X-Ray"))

    # Load selected model
    if model_type == "Brain X-Ray":
        model, class_dict = load_brain_tumor_model()
    else:
        model, labels = load_chest_xray_model()

    # Allow user to upload an image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Make prediction when button is clicked
        if st.button("Predict"):
            if model_type == "Brain X-Ray":
                predicted_class = predict_with_brain_tumor_model(uploaded_image, model)
                st.write(f"Predicted Class: {predicted_class}")
            else:
                predicted_classes = predict_with_chest_xray_model(uploaded_image, model, labels)
                st.write("Predicted Classes:")
                for predicted_class in predicted_classes:
                    st.write(predicted_class)

if __name__ == "__main__":
    main()

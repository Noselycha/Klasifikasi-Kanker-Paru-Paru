import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Path folder model
MODEL_PATHS = {
    "DenseNet": "models/densenet.h5",
    "EfficientNet": "models/efficientnet.h5",
    # Tambahkan model lainnya di sini
}

def load_selected_model(model_name):
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Model {model_name} tidak ditemukan!")
    model_path = MODEL_PATHS[model_name]
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(img_file):
    img = Image.open(img_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
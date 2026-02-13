import os
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# LOAD MODEL
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_model.keras")

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# ----------------------------
# PREPROCESS IMAGE
# ----------------------------
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ----------------------------
# PREDICT
# ----------------------------
def predict(image_path, class_names):
    arr = preprocess_image(image_path)
    preds = model.predict(arr)
    idx = np.argmax(preds[0])
    return {
        "label": class_names[idx],
        "confidence": float(preds[0][idx]),
        "all_probabilities": preds[0].tolist()
    }

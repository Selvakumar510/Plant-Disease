from flask import Flask, request, jsonify
import os
from model_utils import predict

app = Flask(__name__)

# ✅ EXACT class order used during training
CLASS_NAMES = [
    'Healthy_Leaf_Rose',
    'Mango-Healthy',
    'Mango-Powdery Mildew',
    'Mango-SootyMould',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Rose_Rust',
    'Rose_sawfly_Rose_slug',
    'apple-rust',
    'apple-scab',
    'apple_healthy'
]

@app.route("/predict", methods=["POST"])
def detect():
    data = request.json
    image_path = data.get("imagePath")

    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": "Invalid image path"}), 400

    try:
        result = predict(image_path, CLASS_NAMES)
        return jsonify(result)
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    # ✅ AI SERVICE PORT (DO NOT CHANGE)
    app.run(host="127.0.0.1", port=5000, debug=True)

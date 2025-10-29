import os
import io
import json
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify
import tensorflow as tf


MODEL_PATH = "vn_food_model_5class.h5"
LABEL_MAP_PATH = "vn_food_label_map.json"
IMG_SIZE = 150  
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, "r") as f:
    class_names = json.load(f)
app = Flask(__name__)


def prepare_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0  
    arr = np.expand_dims(arr, axis=0)  
    return arr

INGREDIENTS = {
    "pho": [
        "bánh phở",
        "thịt bò/thịt gà",
        "hành lá",
        "nước dùng xương"
    ],
    "banh_mi": [
        "bánh mì",
        "pate/thịt nguội",
        "dưa leo",
        "ngò"
    ],
    "bun_bo_hue": [
        "bún",
        "bò",
        "sả ớt",
        "chả Huế"
    ],
    "com_tam": [
        "cơm hạt gãy",
        "sườn nướng",
        "bì, chả trứng",
        "dưa chua"
    ],
    "goi_cuon": [
        "bánh tráng",
        "tôm",
        "bún",
        "rau sống"
    ]
}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    
    x = prepare_image(img_bytes)

    logits = model.predict(x)  
    pred_index = int(np.argmax(logits, axis=1)[0])
    dish_name = class_names[pred_index]

    result = {
        "prediction": dish_name,
        "ingredients": INGREDIENTS.get(dish_name, [])
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

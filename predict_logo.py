import json, numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (160, 160)

model = tf.keras.models.load_model("crypto_logo_model.h5", compile=False)
labels = json.load(open("labels_logo.json"))

def prep(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)   # don't divide by 255
    return np.expand_dims(x, axis=0)

path = input("Enter image path: ").strip()
x = prep(path)

probs = model.predict(x, verbose=0)[0]
idx = int(np.argmax(probs))

print("\nPrediction:", labels[idx])
for i, name in enumerate(labels):
    print(f"{name}: {probs[i]*100:.2f}%")

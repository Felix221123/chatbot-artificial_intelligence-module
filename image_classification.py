import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import subprocess

IMG_SIZE = (160, 160)

def pick_image_file_mac(prompt="Choose an image to classify"):
    # Native macOS file chooser (no tkinter)
    script = f'POSIX path of (choose file with prompt "{prompt}")'
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if r.returncode != 0:
        return None  # user cancelled
    return r.stdout.strip()

class CryptoLogoClassifier:
    def __init__(self, model_path="crypto_logo_model.h5", labels_path="labels_logo.json"):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        with open(labels_path, "r") as f:
            self.labels = json.load(f)

    def _prep(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        x = np.array(img, dtype=np.float32)   # IMPORTANT: don't divide by 255
        return np.expand_dims(x, axis=0)

    def predict(self, path: str):
        x = self._prep(path)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        return self.labels[idx], float(np.max(probs)), probs

def classify_with_dialog(classifier: CryptoLogoClassifier, open_preview=True) -> str:
    path = pick_image_file_mac()
    if not path:
        return "No image selected."

    if open_preview:
        # Opens the selected image in Preview so you can screenshot alongside terminal output
        subprocess.run(["open", "-a", "Preview", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


    label, conf, probs = classifier.predict(path)
    fname = os.path.basename(path)

    lines = [
        f"Selected image: {fname}",
        f"The image most likely contains: {label} ({conf*100:.1f}%)",
        ""
    ]
    for name, p in zip(classifier.labels, probs):
        lines.append(f"{name}: {p*100:.2f}%")

    return "\n".join(lines)

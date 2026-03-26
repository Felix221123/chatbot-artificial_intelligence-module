import tensorflow as tf

DATA_DIR = "finance_dataset"
IMG_SIZE = (160, 160)
BATCH = 32
CLASSES = ["bitcoin", "ethereum", "solana", "xrp", "litecoin"]

def load_split(split):
    return tf.keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/{split}",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        class_names=CLASSES,
        label_mode="int",
        shuffle=False
    )

test_ds = load_split("test")
model = tf.keras.models.load_model("crypto_logo_model_tuned.h5", compile=False)
model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])

loss, acc = model.evaluate(test_ds, verbose=1)
print(f"\nBASELINE TEST ACCURACY: {acc:.4f}")

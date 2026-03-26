import json
import tensorflow as tf
from tensorflow.keras import layers

data_directory = "finance_dataset"
IMG_SIZE = (160, 160)
batch_size = 32
epochs_head = 10
epochs_fine = 5

CLASSES = ["bitcoin", "ethereum", "solana", "xrp", "litecoin"]

# Load datasets
def load_split(split):
    return tf.keras.utils.image_dataset_from_directory(
        f"{data_directory}/{split}",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="int",
        class_names=CLASSES,
        shuffle=(split == "train"),
    )

train_ds = load_split("train")
val_ds   = load_split("val")
test_ds  = load_split("test")

with open("labels_logo.json", "w") as f:
    json.dump(CLASSES, f)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# Data augmentation & MobileNetV2 base
data_aug = tf.keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.12),
    layers.RandomContrast(0.12),
])

# MobileNetV2 expects inputs in [0,255] range, so we won't rescale by 1/255. Instead, we'll use a Rescaling layer to shift to [-1,1] which is what MobileNetV2 was trained on.
base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

# Build the model
inputs = layers.Input(shape=IMG_SIZE + (3,))
x = data_aug(inputs)
x = layers.Rescaling(1./127.5, offset=-1)(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(CLASSES), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

# Compile and train
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# callbacks for early stopping and saving best model
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best_logo_model.h5", monitor="val_accuracy", save_best_only=True),
]


print("\nTraining head...")
model.fit(train_ds, validation_data=val_ds, epochs=epochs_head, callbacks=callbacks)

print("\nFine-tuning...")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.fit(train_ds, validation_data=val_ds, epochs=epochs_fine, callbacks=callbacks)

test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\nTEST ACCURACY: {test_acc:.4f}")

model.save("crypto_logo_model.h5", include_optimizer=False)
print("Saved crypto_logo_model.h5")

import json
import tensorflow as tf
from tensorflow.keras import layers
import keras_tuner as kt

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
        shuffle=(split == "train"),
    )

train_ds = load_split("train")
val_ds   = load_split("val")
test_ds  = load_split("test")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

def build_model(hp):
    # Tuned hyperparameters
    dropout = hp.Float("dropout", 0.15, 0.50, step=0.05)
    lr_head = hp.Choice("lr_head", [1e-4, 3e-4, 1e-3, 3e-3])

    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False  # tuner searches on head training only

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = layers.RandomRotation(0.08)(inputs)
    x = layers.RandomZoom(0.12)(x)
    x = layers.RandomContrast(0.12)(x)

    # .h5-safe preprocessing
    x = layers.Rescaling(1./127.5, offset=-1)(x)

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_head),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=10,              # good enough for coursework
    executions_per_trial=1,
    directory="tuner_runs",
    project_name="crypto_logo_tuning"
)

tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=6,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2)]
)

best_hp = tuner.get_best_hyperparameters(1)[0]
print("\nBEST HP:")
print(best_hp.values)

with open("best_hyperparams.json", "w") as f:
    json.dump(best_hp.values, f, indent=2)

# ---- Train final model with best hyperparams + fine-tuning depth ----
def train_final(best_hp):
    dropout = best_hp.get("dropout")
    lr_head = best_hp.get("lr_head")

    # Fine-tune hyperparams (keep simple but real)
    fine_layers = 30           # you can change to 10/20/30 and discuss
    lr_fine = 1e-5

    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = layers.RandomRotation(0.08)(inputs)
    x = layers.RandomZoom(0.12)(x)
    x = layers.RandomContrast(0.12)(x)
    x = layers.Rescaling(1./127.5, offset=-1)(x)

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    # Head training
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_head),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(train_ds, validation_data=val_ds, epochs=10,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)])

    # Fine-tuning last N layers
    base.trainable = True
    for layer in base.layers[:-fine_layers]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_fine),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(train_ds, validation_data=val_ds, epochs=6,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)])

    return model

final_model = train_final(best_hp)

loss, acc = final_model.evaluate(test_ds, verbose=1)
print(f"\nTUNED TEST ACCURACY: {acc:.4f}")

final_model.save("crypto_logo_model_tuned.h5", include_optimizer=False)
print("Saved crypto_logo_model_tuned.h5")

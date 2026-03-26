import json
import tensorflow as tf
from tensorflow.keras import layers
import keras_tuner as kt

SEED = 42
tf.keras.utils.set_random_seed(SEED)

DATA_DIR = "finance_dataset"
IMG_SIZE = (160, 160)
BATCH = 16
CLASSES = ["bitcoin", "ethereum", "solana", "xrp", "litecoin"]


def load_split(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/{split}",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        class_names=CLASSES,
        label_mode="int",
        shuffle=shuffle,
        seed=SEED
    )


train_ds = load_split("train", shuffle=True)
val_ds = load_split("val", shuffle=False)
test_ds = load_split("test", shuffle=False)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


def build_model(hp):
    dropout = hp.Float("dropout", 0.10, 0.35, step=0.05)
    lr_head = hp.Choice("lr_head", [1e-4, 3e-4, 1e-3])

    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = layers.RandomRotation(0.04)(inputs)
    x = layers.RandomZoom(0.06)(x)
    x = layers.RandomTranslation(0.04, 0.04)(x)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)

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
    objective=kt.Objective("val_accuracy", direction="max"),
    max_trials=10,
    executions_per_trial=2,   # more stable than 1 on a tiny validation set
    directory="tuner_runs",
    project_name="crypto_logo_tuning"
)

tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
    ],
    verbose=1
)

best_hp = tuner.get_best_hyperparameters(1)[0]
print("\nBEST HP:")
print(best_hp.values)

with open("best_hyperparams.json", "w") as f:
    json.dump(best_hp.values, f, indent=2)


def train_final(best_hp):
    dropout = best_hp.get("dropout")
    lr_head = best_hp.get("lr_head")
    lr_fine = 1e-5
    fine_layers = 40

    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = layers.RandomRotation(0.04)(inputs)
    x = layers.RandomZoom(0.06)(x)
    x = layers.RandomTranslation(0.04, 0.04)(x)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)

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

    head_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=12,
        callbacks=head_callbacks,
        verbose=1
    )

    # Save head-stage best before fine-tuning
    model.save("best_logo_model.h5", include_optimizer=False)

    base.trainable = True
    for layer in base.layers[:-fine_layers]:
        layer.trainable = False

    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_fine),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    fine_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=8,
        callbacks=fine_callbacks,
        verbose=1
    )

    # Keep whichever is better on validation
    head_model = tf.keras.models.load_model("best_logo_model.h5", compile=False)
    head_model.compile(
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    _, head_val_acc = head_model.evaluate(val_ds, verbose=0)
    _, fine_val_acc = model.evaluate(val_ds, verbose=0)

    if head_val_acc >= fine_val_acc:
        print(f"\nKeeping head-stage model (val_acc={head_val_acc:.4f}) over fine-tuned model (val_acc={fine_val_acc:.4f})")
        return head_model

    print(f"\nKeeping fine-tuned model (val_acc={fine_val_acc:.4f}) over head-stage model (val_acc={head_val_acc:.4f})")
    return model


final_model = train_final(best_hp)

loss, acc = final_model.evaluate(test_ds, verbose=1)
print(f"\nTUNED TEST ACCURACY: {acc:.4f}")

final_model.save("crypto_logo_model_tuned.h5", include_optimizer=False)
print("Saved crypto_logo_model_tuned.h5")

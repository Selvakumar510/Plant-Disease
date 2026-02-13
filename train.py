import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# ----------------------------
# PATHS
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "..", "backend", "data_split", "train")
VAL_DIR   = os.path.join(BASE_DIR, "..", "backend", "data_split", "val")
TEST_DIR  = os.path.join(BASE_DIR, "..", "backend", "data_split", "test")

# ----------------------------
# PARAMETERS
# ----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# ----------------------------
# DATA AUGMENTATION
# ----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(0.2),
])

# ----------------------------
# LOAD DATA
# ----------------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ----------------------------
# MODEL (MobileNetV2)
# ----------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------
# TRAIN + SAFE SAVE
# ----------------------------
history = None

try:
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
except KeyboardInterrupt:
    print("⚠️ Training interrupted manually")
finally:
    os.makedirs("models", exist_ok=True)
    model.save("models/plant_model.keras")
    print("✅ Model saved to models/plant_model.keras")

# ----------------------------
# PLOT ACCURACY (ONLY IF AVAILABLE)
# ----------------------------
if history is not None:
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.show()
else:
    print("ℹ️ Training history not available (stopped early). Plot skipped.")

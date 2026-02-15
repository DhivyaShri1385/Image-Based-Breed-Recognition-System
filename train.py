import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models

# ==============================
# SETTINGS
# ==============================
IMG_SIZE = 256
BATCH_SIZE = 32
INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 7

train_dir = "dataset/train"
valid_dir = "dataset/valid"

# ==============================
# DATA AUGMENTATION (Stronger)
# ==============================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)

valid_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_data = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("Class indices:", train_data.class_indices)

with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

# ==============================
# BUILD MODEL
# ==============================
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # Phase 1

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_data.num_classes, activation='softmax')
])

# ==============================
# PHASE 1 - FEATURE EXTRACTION
# ==============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

print("\n===== PHASE 1: TRAINING TOP LAYERS =====\n")

history1 = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=INITIAL_EPOCHS,
    callbacks=[early_stop]
)

# ==============================
# PHASE 2 - FINE TUNING
# ==============================
print("\n===== PHASE 2: FINE-TUNING =====\n")

base_model.trainable = True

# Freeze lower layers
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[early_stop]
)

# ==============================
# SAVE MODEL
# ==============================
os.makedirs("model", exist_ok=True)
model.save("model/best_model.keras")

print("\nModel saved successfully as model/best_model.keras")

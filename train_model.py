import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, LeakyReLU, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('models', exist_ok=True)

# ── Image generators ──────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# ── Training data ─────────────────────────────────────────────────────────────
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    shuffle=True
)

# ── Validation data ───────────────────────────────────────────────────────────
validation_generator = validation_datagen.flow_from_directory(
    'dataset/test',
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())
print(f"Classes: {class_names}")

# ── Compute class weights for imbalanced dataset ──────────────────────────────
class_weights_array = compute_class_weight(
    'balanced',
    classes=np.arange(len(class_names)),
    y=train_generator.classes
)
class_weights = {i: w for i, w in enumerate(class_weights_array)}
print(f"Class weights: {class_weights}")

# ── CNN Model ─────────────────────────────────────────────────────────────────
model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), padding='same', input_shape=(48, 48, 1)),
    BatchNormalization(),
    LeakyReLU(negative_slope=0.1),
    Conv2D(32, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(negative_slope=0.1),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    # Block 2
    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(negative_slope=0.1),
    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(negative_slope=0.1),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(negative_slope=0.1),
    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(negative_slope=0.1),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    # Block 4
    Conv2D(256, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(negative_slope=0.1),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.35),

    GlobalAveragePooling2D(),

    Dense(256, kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    LeakyReLU(negative_slope=0.1),
    Dropout(0.5),

    Dense(128, kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    LeakyReLU(negative_slope=0.1),
    Dropout(0.4),

    Dense(7, activation='softmax')
])

model.summary()

# ── Compile ───────────────────────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ── Callbacks ─────────────────────────────────────────────────────────────────
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'models/emotion_model_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# ── Train ─────────────────────────────────────────────────────────────────────
NUM_EPOCHS = 30

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    class_weight=class_weights,
    callbacks=[reduce_lr, checkpoint]
)

# ── Save final model ──────────────────────────────────────────────────────────
model.save('models/emotion_model.h5')
print("Model saved to models/emotion_model.h5")

# ── Convert to TFLite ─────────────────────────────────────────────────────────
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('models/emotion_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("TFLite model saved to models/emotion_model.tflite")

# ── Evaluate on test set ──────────────────────────────────────────────────────
print("\n--- Evaluation on Test Set ---")
validation_generator.reset()
test_loss, test_acc = model.evaluate(
    validation_generator,
    steps=validation_generator.samples // validation_generator.batch_size
)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# ── Plot training history ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/training_history.png', dpi=150)
plt.show()
print("Training plot saved to models/training_history.png")

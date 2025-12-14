"""
Skin Disease Detection Model
CNN-based deep learning model for classifying skin diseases
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D, Input
)
from tensorflow.keras.applications import MobileNetV2, ResNet50V2
from tensorflow.keras.optimizers import Adam


# Default skin disease classes (can be customized based on dataset)
DEFAULT_CLASSES = [
    'Acne',
    'Eczema',
    'Melanoma',
    'Psoriasis',
    'Tinea (Ringworm)',
    'Vitiligo',
    'Normal Skin'
]

IMG_SIZE = 224


def create_simple_cnn(num_classes=7, input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """
    Create a simple CNN model for skin disease classification.
    Good for smaller datasets and faster training.
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Fourth Convolutional Block
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Fully Connected Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model


def create_mobilenet_model(num_classes=7, input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """
    Create a MobileNetV2-based model using transfer learning.
    Lightweight and efficient for mobile/web deployment.
    """
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze base model layers
    base_model.trainable = False

    # Build the model
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    return model, base_model


def create_resnet_model(num_classes=7, input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """
    Create a ResNet50V2-based model using transfer learning.
    Higher accuracy but more computationally intensive.
    """
    # Load pre-trained ResNet50V2
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze base model layers
    base_model.trainable = False

    # Build the model
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    return model, base_model


def compile_model(model, learning_rate=0.001):
    """Compile the model with Adam optimizer and categorical crossentropy loss."""
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def unfreeze_base_model(base_model, num_layers_to_unfreeze=20):
    """
    Unfreeze the last n layers of the base model for fine-tuning.
    Call this after initial training to improve accuracy.
    """
    base_model.trainable = True

    # Freeze all layers except the last n
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    return base_model


def get_model_summary(model):
    """Print model summary."""
    return model.summary()


if __name__ == "__main__":
    # Test model creation
    print("Testing Simple CNN Model:")
    simple_model = create_simple_cnn(num_classes=7)
    simple_model = compile_model(simple_model)
    simple_model.summary()

    print("\n" + "="*50 + "\n")

    print("Testing MobileNetV2 Model:")
    mobilenet_model, _ = create_mobilenet_model(num_classes=7)
    mobilenet_model = compile_model(mobilenet_model)
    mobilenet_model.summary()

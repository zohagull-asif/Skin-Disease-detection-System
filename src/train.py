"""
Training Script for Skin Disease Detection Model
"""

import os
import sys
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import (
    create_simple_cnn, create_mobilenet_model, create_resnet_model,
    compile_model, unfreeze_base_model
)
from src.data_preprocessing import (
    create_data_generators, get_class_weights, load_dataset_info
)


def setup_gpu():
    """Configure GPU settings for training."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found. Training on CPU.")


def create_callbacks(model_save_path, log_dir):
    """Create training callbacks."""
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),

        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),

        # TensorBoard logging
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        ),

        # CSV logging
        CSVLogger(
            os.path.join(log_dir, 'training_log.csv'),
            separator=',',
            append=False
        )
    ]

    return callbacks


def plot_training_history(history, save_path):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to: {save_path}")


def train_model(
    data_dir,
    model_type='mobilenet',
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    fine_tune=True,
    fine_tune_epochs=20
):
    """
    Train the skin disease detection model.

    Args:
        data_dir: Directory containing training data
        model_type: Type of model ('simple', 'mobilenet', 'resnet')
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        fine_tune: Whether to fine-tune the model (for transfer learning models)
        fine_tune_epochs: Number of fine-tuning epochs
    """
    # Setup
    setup_gpu()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directories
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    log_dir = os.path.join(output_dir, 'logs', timestamp)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print("SKIN DISEASE DETECTION - MODEL TRAINING")
    print("=" * 60)

    # Load dataset info
    print("\nDataset Information:")
    print("-" * 40)
    stats = load_dataset_info(data_dir)

    if stats.get('total', 0) == 0:
        print("\nERROR: No images found in the dataset directory!")
        print(f"Please add images to: {data_dir}")
        print("\nExpected structure:")
        print("  data/train/")
        print("    ├── Acne/")
        print("    ├── Eczema/")
        print("    ├── Melanoma/")
        print("    └── ... (one folder per class)")
        return None

    # Create data generators
    print("\nCreating data generators...")
    train_generator, val_generator = create_data_generators(
        data_dir,
        validation_split=0.2,
        batch_size=batch_size
    )

    num_classes = train_generator.num_classes
    class_names = list(train_generator.class_indices.keys())
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")

    # Calculate class weights for imbalanced data
    class_weights = get_class_weights(train_generator)
    print(f"Class weights: {class_weights}")

    # Create model
    print(f"\nCreating {model_type} model...")
    base_model = None

    if model_type == 'simple':
        model = create_simple_cnn(num_classes=num_classes)
    elif model_type == 'mobilenet':
        model, base_model = create_mobilenet_model(num_classes=num_classes)
    elif model_type == 'resnet':
        model, base_model = create_resnet_model(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Compile model
    model = compile_model(model, learning_rate=learning_rate)
    print("\nModel Summary:")
    model.summary()

    # Model save path
    model_save_path = os.path.join(output_dir, f'skin_disease_model_{model_type}_{timestamp}.keras')

    # Create callbacks
    callbacks = create_callbacks(model_save_path, log_dir)

    # Initial training
    print("\n" + "=" * 60)
    print("PHASE 1: Initial Training")
    print("=" * 60)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Fine-tuning (for transfer learning models)
    if fine_tune and base_model is not None:
        print("\n" + "=" * 60)
        print("PHASE 2: Fine-tuning")
        print("=" * 60)

        # Unfreeze some layers of the base model
        base_model = unfreeze_base_model(base_model, num_layers_to_unfreeze=30)

        # Recompile with lower learning rate
        model = compile_model(model, learning_rate=learning_rate / 10)

        # Update callbacks for fine-tuning
        fine_tune_callbacks = create_callbacks(
            model_save_path.replace('.keras', '_finetuned.keras'),
            os.path.join(log_dir, 'fine_tuning')
        )

        # Fine-tune
        history_fine = model.fit(
            train_generator,
            epochs=fine_tune_epochs,
            validation_data=val_generator,
            callbacks=fine_tune_callbacks,
            class_weight=class_weights,
            verbose=1
        )

        # Combine histories
        for key in history.history:
            history.history[key].extend(history_fine.history[key])

    # Plot training history
    plot_path = os.path.join(log_dir, 'training_history.png')
    plot_training_history(history, plot_path)

    # Save class names
    class_names_path = os.path.join(output_dir, 'class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)
    print(f"\nClass names saved to: {class_names_path}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model saved to: {model_save_path}")
    print(f"Training logs saved to: {log_dir}")

    # Print final metrics
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")

    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train Skin Disease Detection Model')

    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/train',
        help='Directory containing training data'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='mobilenet',
        choices=['simple', 'mobilenet', 'resnet'],
        help='Type of model to train'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--no_fine_tune',
        action='store_true',
        help='Disable fine-tuning for transfer learning models'
    )
    parser.add_argument(
        '--fine_tune_epochs',
        type=int,
        default=20,
        help='Number of fine-tuning epochs'
    )

    args = parser.parse_args()

    # Get absolute path for data directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, args.data_dir)

    # Train model
    train_model(
        data_dir=data_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fine_tune=not args.no_fine_tune,
        fine_tune_epochs=args.fine_tune_epochs
    )


if __name__ == "__main__":
    main()

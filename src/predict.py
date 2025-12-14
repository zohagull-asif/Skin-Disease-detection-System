"""
Prediction/Inference Module for Skin Disease Detection
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import tensorflow as tf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import preprocess_single_image, preprocess_uploaded_image


# Default class names (update based on your trained model)
DEFAULT_CLASS_NAMES = [
    'Acne',
    'Eczema',
    'Melanoma',
    'Psoriasis',
    'Tinea (Ringworm)',
    'Vitiligo',
    'Normal Skin'
]

# Disease information for each class
DISEASE_INFO = {
    'Acne': {
        'description': 'Acne is a skin condition that occurs when hair follicles become plugged with oil and dead skin cells.',
        'severity': 'Mild to Moderate',
        'recommendations': [
            'Keep skin clean with gentle cleansers',
            'Avoid touching your face',
            'Use non-comedogenic products',
            'Consult a dermatologist for persistent acne'
        ]
    },
    'Eczema': {
        'description': 'Eczema (atopic dermatitis) is a condition that makes skin red, inflamed, and itchy.',
        'severity': 'Mild to Severe',
        'recommendations': [
            'Moisturize regularly',
            'Avoid triggers like certain soaps and allergens',
            'Use prescribed topical treatments',
            'Consult a dermatologist for management'
        ]
    },
    'Melanoma': {
        'description': 'Melanoma is the most serious type of skin cancer that develops in melanocytes.',
        'severity': 'Serious - Requires Immediate Attention',
        'recommendations': [
            'IMMEDIATELY consult a dermatologist or oncologist',
            'Do not delay medical evaluation',
            'Regular skin checks are essential',
            'Protect skin from UV exposure'
        ]
    },
    'Psoriasis': {
        'description': 'Psoriasis is a skin disease that causes red, itchy scaly patches.',
        'severity': 'Moderate',
        'recommendations': [
            'Use prescribed topical treatments',
            'Moisturize regularly',
            'Phototherapy may help',
            'Consult a dermatologist for treatment plan'
        ]
    },
    'Tinea (Ringworm)': {
        'description': 'Ringworm is a fungal infection that causes a ring-shaped rash on the skin.',
        'severity': 'Mild',
        'recommendations': [
            'Use antifungal creams or medications',
            'Keep the affected area clean and dry',
            'Avoid sharing personal items',
            'Consult a doctor if it persists'
        ]
    },
    'Vitiligo': {
        'description': 'Vitiligo is a condition where patches of skin lose their pigment.',
        'severity': 'Mild (Cosmetic)',
        'recommendations': [
            'Protect affected areas from sun',
            'Consider cosmetic options if desired',
            'Phototherapy may help in some cases',
            'Consult a dermatologist for treatment options'
        ]
    },
    'Normal Skin': {
        'description': 'The skin appears healthy with no visible conditions.',
        'severity': 'None',
        'recommendations': [
            'Maintain good skincare routine',
            'Use sunscreen regularly',
            'Stay hydrated',
            'Regular skin check-ups'
        ]
    }
}


class SkinDiseasePredictor:
    """Class for making predictions on skin disease images."""

    def __init__(self, model_path=None, class_names=None):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the trained model file
            class_names: List of class names (or path to JSON file)
        """
        self.model = None
        self.class_names = class_names or DEFAULT_CLASS_NAMES

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

        # Load class names from JSON if path provided
        if isinstance(class_names, str) and os.path.exists(class_names):
            with open(class_names, 'r') as f:
                self.class_names = json.load(f)

    def load_model(self, model_path):
        """Load a trained model from file."""
        print(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return self.model

    def predict(self, image_path):
        """
        Make a prediction on a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Preprocess image
        img_array = preprocess_single_image(image_path)

        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        # Get class name
        predicted_class = self.class_names[predicted_class_idx]

        # Get all class probabilities
        all_probabilities = {
            self.class_names[i]: float(predictions[0][i])
            for i in range(len(self.class_names))
        }

        # Sort by probability
        all_probabilities = dict(
            sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)
        )

        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probabilities,
            'disease_info': DISEASE_INFO.get(predicted_class, {})
        }

        return result

    def predict_from_array(self, img_array):
        """
        Make a prediction on a preprocessed image array.

        Args:
            img_array: Preprocessed image array

        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        # Get class name
        predicted_class = self.class_names[predicted_class_idx]

        # Get all class probabilities
        all_probabilities = {
            self.class_names[i]: float(predictions[0][i])
            for i in range(len(self.class_names))
        }

        # Sort by probability
        all_probabilities = dict(
            sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)
        )

        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probabilities,
            'disease_info': DISEASE_INFO.get(predicted_class, {})
        }

        return result

    def predict_batch(self, image_paths):
        """
        Make predictions on multiple images.

        Args:
            image_paths: List of paths to image files

        Returns:
            List of prediction results
        """
        results = []
        for path in image_paths:
            try:
                result = self.predict(path)
                result['image_path'] = path
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': path,
                    'error': str(e)
                })
        return results


def print_prediction(result):
    """Pretty print a prediction result."""
    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)

    print(f"\nPredicted Condition: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")

    if result.get('disease_info'):
        info = result['disease_info']
        print(f"\nSeverity: {info.get('severity', 'Unknown')}")
        print(f"\nDescription: {info.get('description', 'N/A')}")

        if info.get('recommendations'):
            print("\nRecommendations:")
            for rec in info['recommendations']:
                print(f"  - {rec}")

    print("\nAll Probabilities:")
    for class_name, prob in result['all_probabilities'].items():
        bar = "█" * int(prob * 20)
        print(f"  {class_name:20s}: {prob*100:5.2f}% {bar}")

    print("\n" + "=" * 50)
    print("DISCLAIMER: This is an AI-based prediction and should")
    print("NOT be used as a substitute for professional medical advice.")
    print("Please consult a dermatologist for accurate diagnosis.")
    print("=" * 50)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Skin Disease Prediction')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to the image file for prediction'
    )
    parser.add_argument(
        '--class_names',
        type=str,
        default=None,
        help='Path to class names JSON file'
    )

    args = parser.parse_args()

    # Create predictor
    predictor = SkinDiseasePredictor(
        model_path=args.model,
        class_names=args.class_names
    )

    # Make prediction
    result = predictor.predict(args.image)

    # Print result
    print_prediction(result)


if __name__ == "__main__":
    main()

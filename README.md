# Skin Disease Detection System

An AI-powered skin disease detection system using Deep Learning (CNN) with a web-based interface.

## Features

- **Deep Learning Models**: Multiple CNN architectures (Simple CNN, MobileNetV2, ResNet50V2)
- **Transfer Learning**: Pre-trained models for better accuracy with limited data
- **Web Interface**: User-friendly Streamlit application
- **Data Augmentation**: Robust training with image augmentation
- **Multi-class Classification**: Detects multiple skin conditions

## Supported Skin Conditions

- Acne
- Eczema
- Melanoma
- Psoriasis
- Tinea (Ringworm)
- Vitiligo
- Normal Skin

## Project Structure

```
skin disease detection system/
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── src/
│   ├── model.py             # CNN model architectures
│   ├── data_preprocessing.py # Data processing utilities
│   ├── train.py             # Training script
│   └── predict.py           # Prediction/inference module
├── data/
│   ├── train/               # Training images (organized by class)
│   └── test/                # Test images
├── models/                   # Saved trained models
└── static/                   # Static files
```

## Installation

1. **Clone or download this project**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

Organize your training images in the following structure:

```
data/train/
├── Acne/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Eczema/
│   ├── image1.jpg
│   └── ...
├── Melanoma/
│   └── ...
├── Psoriasis/
│   └── ...
├── Tinea/
│   └── ...
├── Vitiligo/
│   └── ...
└── Normal/
    └── ...
```

### Recommended Datasets

You can download skin disease datasets from:
- [ISIC Archive](https://www.isic-archive.com/) - Dermoscopy images
- [DermNet](https://dermnetnz.org/) - Clinical images
- [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) - Skin lesion dataset
- [Kaggle Skin Disease Datasets](https://www.kaggle.com/datasets?search=skin+disease)

## Training the Model

### Basic Training

```bash
python src/train.py --data_dir data/train --model_type mobilenet --epochs 50
```

### Training Options

```bash
python src/train.py \
    --data_dir data/train \
    --model_type mobilenet \     # Options: simple, mobilenet, resnet
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --fine_tune_epochs 20
```

### Model Types

| Model Type | Description | Use Case |
|------------|-------------|----------|
| `simple` | Custom CNN | Small datasets, fast training |
| `mobilenet` | MobileNetV2 | Balanced accuracy/speed, deployment |
| `resnet` | ResNet50V2 | Higher accuracy, more compute |

## Running the Web Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Making Predictions via Command Line

```bash
python src/predict.py --model models/your_model.keras --image path/to/image.jpg
```

## API Usage

```python
from src.predict import SkinDiseasePredictor

# Initialize predictor
predictor = SkinDiseasePredictor(model_path='models/your_model.keras')

# Make prediction
result = predictor.predict('path/to/skin_image.jpg')

print(f"Condition: {result['predicted_class']}")
print(f"Confidence: {result['confidence']*100:.2f}%")
```

## Model Performance Tips

1. **Data Quality**: Use clear, well-lit images
2. **Data Quantity**: Aim for at least 100+ images per class
3. **Balanced Dataset**: Try to have similar number of images per class
4. **Augmentation**: The training script includes extensive augmentation
5. **Fine-tuning**: Transfer learning models are fine-tuned for better accuracy

## Technical Details

### Model Architecture

- **Input**: 224x224 RGB images
- **Backbone**: MobileNetV2/ResNet50V2 (pre-trained on ImageNet)
- **Head**: Global Average Pooling + Dense layers with dropout
- **Output**: Softmax probabilities for each class

### Training Features

- Data augmentation (rotation, flip, zoom, brightness)
- Class weight balancing for imbalanced datasets
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping to prevent overfitting
- Model checkpointing to save best model

## Disclaimer

**IMPORTANT**: This system is for educational and research purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any skin-related concerns.

## License

This project is for educational purposes.

## Acknowledgments

- TensorFlow/Keras for deep learning framework
- Streamlit for web interface
- ImageNet for pre-trained weights

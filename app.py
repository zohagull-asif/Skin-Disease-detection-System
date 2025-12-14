"""
Skin Disease Detection System - Web Interface
Streamlit-based web application for skin disease classification
"""

import os
import sys
import json
import streamlit as st
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Add source directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import preprocess_uploaded_image
from src.predict import SkinDiseasePredictor, DISEASE_INFO, DEFAULT_CLASS_NAMES

# Page configuration
st.set_page_config(
    page_title="Skin Disease Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #17a2b8;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def load_model():
    """Load the trained model."""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')

    # Look for model files
    model_files = []
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith(('.keras', '.h5'))]

    if not model_files:
        return None, None

    # Use the most recent model
    model_path = os.path.join(models_dir, sorted(model_files)[-1])

    # Load class names if available
    class_names_path = os.path.join(models_dir, 'class_names.json')
    class_names = None
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)

    predictor = SkinDiseasePredictor(model_path, class_names)
    return predictor, model_path


def create_probability_chart(probabilities):
    """Create a bar chart of prediction probabilities."""
    classes = list(probabilities.keys())
    probs = [p * 100 for p in probabilities.values()]

    colors = ['#1E88E5' if i == 0 else '#90CAF9' for i in range(len(classes))]

    fig = go.Figure(data=[
        go.Bar(
            x=probs,
            y=classes,
            orientation='h',
            marker_color=colors,
            text=[f'{p:.1f}%' for p in probs],
            textposition='auto'
        )
    ])

    fig.update_layout(
        title='Prediction Probabilities',
        xaxis_title='Probability (%)',
        yaxis_title='Condition',
        height=400,
        margin=dict(l=150, r=50, t=50, b=50)
    )

    return fig


def get_confidence_class(confidence):
    """Get CSS class based on confidence level."""
    if confidence >= 0.7:
        return 'confidence-high'
    elif confidence >= 0.4:
        return 'confidence-medium'
    else:
        return 'confidence-low'


def display_disease_info(disease_name, disease_info):
    """Display information about the detected disease."""
    st.subheader(f"About {disease_name}")

    if disease_info:
        st.write(f"**Description:** {disease_info.get('description', 'N/A')}")

        severity = disease_info.get('severity', 'Unknown')
        if 'Serious' in severity or 'Immediate' in severity:
            st.markdown(f'<div class="danger-box"><strong>Severity:</strong> {severity}</div>',
                       unsafe_allow_html=True)
        elif 'Moderate' in severity:
            st.markdown(f'<div class="warning-box"><strong>Severity:</strong> {severity}</div>',
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="info-box"><strong>Severity:</strong> {severity}</div>',
                       unsafe_allow_html=True)

        recommendations = disease_info.get('recommendations', [])
        if recommendations:
            st.write("**Recommendations:**")
            for rec in recommendations:
                st.write(f"• {rec}")


def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Skin Disease Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Skin Condition Analysis using Deep Learning</p>',
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This application uses deep learning to analyze skin images
        and detect potential skin conditions.

        **Supported Conditions:**
        - Acne
        - Eczema
        - Melanoma
        - Psoriasis
        - Ringworm (Tinea)
        - Vitiligo
        - Normal Skin
        """)

        st.divider()

        st.header("⚠️ Disclaimer")
        st.warning("""
        This tool is for educational purposes only and should NOT
        be used as a substitute for professional medical advice,
        diagnosis, or treatment. Always consult a qualified
        healthcare provider for skin-related concerns.
        """)

        st.divider()

        st.header("📊 Model Info")
        predictor, model_path = load_model()
        if predictor and predictor.model:
            st.success("Model loaded successfully!")
            st.write(f"**Model:** {os.path.basename(model_path) if model_path else 'N/A'}")
        else:
            st.error("No trained model found!")
            st.info("Please train a model first using the training script.")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Image")

        uploaded_file = st.file_uploader(
            "Choose a skin image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a clear image of the skin area you want to analyze"
        )

        if uploaded_file:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Image info
            st.write(f"**Image Size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Format:** {image.format or 'Unknown'}")

    with col2:
        st.header("🔍 Analysis Results")

        if uploaded_file:
            # Load model if not already loaded
            if 'predictor' not in st.session_state or st.session_state.predictor is None:
                predictor, model_path = load_model()
                st.session_state.predictor = predictor
            else:
                predictor = st.session_state.predictor

            if predictor and predictor.model:
                with st.spinner("Analyzing image..."):
                    # Preprocess and predict
                    img_array, _ = preprocess_uploaded_image(uploaded_file)
                    result = predictor.predict_from_array(img_array)

                # Display results
                predicted_class = result['predicted_class']
                confidence = result['confidence']

                # Main prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Detected Condition: {predicted_class}</h3>
                    <p class="{get_confidence_class(confidence)}">
                        Confidence: {confidence*100:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Warning for serious conditions
                if predicted_class == 'Melanoma':
                    st.error("""
                    ⚠️ **IMPORTANT:** Melanoma is a serious condition.
                    Please consult a dermatologist immediately for proper evaluation.
                    """)

                # Probability chart
                st.plotly_chart(
                    create_probability_chart(result['all_probabilities']),
                    use_container_width=True
                )

                # Disease information
                display_disease_info(
                    predicted_class,
                    result.get('disease_info', DISEASE_INFO.get(predicted_class, {}))
                )

            else:
                st.warning("""
                ⚠️ **No trained model available!**

                To use this application:
                1. Prepare your dataset in `data/train/` folder
                2. Run the training script: `python src/train.py`
                3. Reload this application

                For demo purposes, a sample prediction interface is shown below.
                """)

                # Demo mode
                st.info("**Demo Mode:** Showing sample prediction interface")

                # Sample probabilities for demo
                demo_probs = {
                    'Normal Skin': 0.45,
                    'Acne': 0.25,
                    'Eczema': 0.15,
                    'Psoriasis': 0.08,
                    'Melanoma': 0.04,
                    'Vitiligo': 0.02,
                    'Tinea (Ringworm)': 0.01
                }

                st.plotly_chart(
                    create_probability_chart(demo_probs),
                    use_container_width=True
                )
        else:
            st.info("👆 Please upload an image to start the analysis")

            # Show sample images info
            st.subheader("📸 Tips for Best Results")
            st.write("""
            For accurate predictions:
            - Use a clear, well-lit image
            - Focus on the affected skin area
            - Avoid blurry or dark images
            - Include only the skin area of interest
            - Use images with adequate resolution
            """)

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Skin Disease Detection System | Powered by Deep Learning</p>
        <p><small>This is an educational tool. Always consult a healthcare professional for medical advice.</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

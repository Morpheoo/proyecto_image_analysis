"""
Streamlit application for fruit quality classification.
Interactive interface for segmentation and quality evaluation.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np
from PIL import Image
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation import segment_image
from src.train import load_model
from src.dataset import get_transforms, QUALITY_NAMES
from src.utils import get_device


# Page configuration
st.set_page_config(
    page_title="🍎 Fruit Quality Analyzer",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #48dbfb;
    }
    .prediction-fresh {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
    }
    .prediction-rotten {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classification_model(model_path: str, model_name: str = "mobilenetv2"):
    """Load and cache the classification model."""
    try:
        model = load_model(model_path, model_name=model_name)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def predict_quality(model, image: np.ndarray, device: torch.device) -> dict:
    """
    Predict fruit quality from image.
    
    Returns:
        Dictionary with prediction, confidence, and probabilities
    """
    transform = get_transforms("test")
    
    # Convert to PIL and apply transforms
    pil_image = Image.fromarray(image)
    img_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    # Inference
    model = model.to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, pred = outputs.max(1)
    
    pred_label = pred.item()
    confidence = probs[0][pred_label].item()
    
    return {
        'prediction': QUALITY_NAMES[pred_label],
        'label_idx': pred_label,
        'confidence': confidence,
        'probabilities': {
            QUALITY_NAMES[i]: float(probs[0][i]) 
            for i in range(len(QUALITY_NAMES))
        }
    }


def save_evaluation(image, segmented, result, seg_method, output_dir):
    """Save evaluation results to streamlit_samples folder."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{result['prediction'].lower()}_{timestamp}"
    
    # Save original
    Image.fromarray(image).save(os.path.join(output_dir, f"{prefix}_original.png"))
    
    # Save segmented
    if segmented is not None:
        Image.fromarray(segmented).save(os.path.join(output_dir, f"{prefix}_segmented.png"))
    
    # Save result info
    import json
    result_info = {
        'prediction': result['prediction'],
        'confidence': result['confidence'],
        'probabilities': result['probabilities'],
        'segmentation_method': seg_method,
        'timestamp': timestamp
    }
    with open(os.path.join(output_dir, f"{prefix}_result.json"), 'w') as f:
        json.dump(result_info, f, indent=2)
    
    return prefix


def main():
    # Header
    st.markdown('<h1 class="main-header">🍎 Fruit Quality Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("Classification Model")
        model_type = st.selectbox(
            "Model Architecture",
            ["mobilenetv2"],
            help="Pre-trained model for classification"
        )
        
        use_segmented_model = st.checkbox(
            "Use segmentation-trained model",
            value=True,
            help="Model trained with segmented images"
        )
        
        model_suffix = "_segmented" if use_segmented_model else "_baseline"
        model_path = PROJECT_ROOT / "models" / f"fruit_quality{model_suffix}.pth"
        
        # Segmentation settings
        st.subheader("🔍 Segmentation")
        seg_method = st.selectbox(
            "Method",
            ["GrabCut", "HSV + Morphology"],
            help="Classical segmentation technique"
        )
        
        if seg_method == "GrabCut":
            iterations = st.slider("Iterations", 1, 10, 5)
            margin = st.slider("Margin", 5, 30, 10)
        else:
            kernel_size = st.slider("Kernel Size", 3, 11, 5, step=2)
        
        # Info panel
        st.markdown("---")
        st.subheader("📋 Info Panel")
        st.markdown(f"""
        <div class="info-box">
            <b>Segmentation:</b> {seg_method}<br>
            <b>Model:</b> MobileNetV2<br>
            <b>Training:</b> {'With segmentation' if use_segmented_model else 'Baseline'}<br>
            <b>Preprocessing:</b> Resize 224×224, ImageNet normalization
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a fruit image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a JPG or PNG image of a fruit"
        )
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file).convert('RGB')
            image_np = np.array(image)
            
            st.image(image, caption="Original Image", use_container_width=True)
    
    with col2:
        if uploaded_file is not None:
            st.subheader("🔬 Analysis Results")
            
            # Apply segmentation
            with st.spinner("Applying segmentation..."):
                try:
                    if seg_method == "GrabCut":
                        seg_result = segment_image(
                            image_np, 
                            method="grabcut",
                            iterations=iterations,
                            margin=margin
                        )
                    else:
                        seg_result = segment_image(
                            image_np, 
                            method="hsv",
                            kernel_size=kernel_size
                        )
                    
                    segmented = seg_result['segmented']
                    mask = seg_result['mask']
                    method_info = seg_result['method_info']
                    
                except Exception as e:
                    st.error(f"Segmentation failed: {e}")
                    segmented = image_np
                    mask = np.ones(image_np.shape[:2], dtype=np.uint8) * 255
                    method_info = "Fallback (no segmentation)"
            
            # Display segmentation results
            seg_col1, seg_col2, seg_col3 = st.columns(3)
            
            with seg_col1:
                st.image(image_np, caption="Original", use_container_width=True)
            
            with seg_col2:
                # Display mask with colormap
                mask_display = np.stack([mask, mask, mask], axis=2)
                st.image(mask_display, caption="Segmentation Mask", use_container_width=True)
            
            with seg_col3:
                st.image(segmented, caption="Segmented", use_container_width=True)
            
            st.info(f"**Segmentation Method:** {method_info}")
            
            # Evaluation button
            st.markdown("---")
            
            if st.button("🔮 Evaluate Quality", type="primary", use_container_width=True):
                # Check model exists
                if not model_path.exists():
                    st.error(f"""
                    Model not found at: {model_path}
                    
                    Please train the model first by running:
                    ```
                    python main.py
                    ```
                    """)
                else:
                    with st.spinner("Analyzing fruit quality..."):
                        # Load model
                        model = load_classification_model(str(model_path))
                        
                        if model is not None:
                            device = get_device()
                            
                            # Get prediction
                            result = predict_quality(model, segmented, device)
                            
                            # Display results
                            st.markdown("### 📊 Prediction Results")
                            
                            pred_class = "prediction-fresh" if result['prediction'] == 'Fresh' else "prediction-rotten"
                            emoji = "✅" if result['prediction'] == 'Fresh' else "⚠️"
                            
                            st.markdown(f"""
                            <div class="{pred_class}">
                                {emoji} <b>{result['prediction']}</b><br>
                                Confidence: {result['confidence']*100:.1f}%
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Probability breakdown
                            st.markdown("#### Probability Distribution")
                            prob_col1, prob_col2 = st.columns(2)
                            
                            with prob_col1:
                                st.metric(
                                    "🍏 Fresh",
                                    f"{result['probabilities']['Fresh']*100:.1f}%"
                                )
                            
                            with prob_col2:
                                st.metric(
                                    "🍂 Rotten",
                                    f"{result['probabilities']['Rotten']*100:.1f}%"
                                )
                            
                            # Progress bar
                            st.progress(result['probabilities']['Fresh'])
                            
                            # Save evaluation
                            output_dir = PROJECT_ROOT / "outputs" / "streamlit_samples"
                            saved = save_evaluation(
                                image_np, segmented, result, 
                                seg_method, str(output_dir)
                            )
                            st.success(f"Evaluation saved: {saved}")
        
        else:
            st.info("👈 Upload an image to start the analysis")
            
            # Show sample info
            st.markdown("""
            ### How to use:
            1. **Upload** a fruit image (apple, banana, or orange)
            2. **Select** segmentation method in the sidebar
            3. **Click** "Evaluate Quality" to get the prediction
            
            ### Supported fruits:
            - 🍎 Apples (fresh & rotten)
            - 🍌 Bananas (fresh & rotten)  
            - 🍊 Oranges (fresh & rotten)
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888;">
        Fruit Quality Analyzer | Image Analysis Project | 2026
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

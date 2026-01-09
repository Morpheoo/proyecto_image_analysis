"""
Aplicación Streamlit para clasificación de calidad de frutas.
Interfaz interactiva para segmentación y evaluación de calidad.

Modo Demo: Solo requiere el modelo entrenado, no necesita dataset.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np
from PIL import Image
import torch
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation import segment_image
from src.inference import load_inference_model, predict, check_model_status, get_device
from src.config import CLASS_NAMES, DEFAULT_MODEL_PATH, SEGMENTATION_METHODS


# Page configuration
st.set_page_config(
    page_title="🍎 Analizador de Calidad de Frutas",
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
    .fruit-type-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-top: 10px;
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


def detect_fruit_type(image: np.ndarray) -> dict:
    """
    Detecta el tipo de fruta basándose en análisis de color.
    
    Args:
        image: Imagen RGB como numpy array
        
    Returns:
        Dictionary con tipo de fruta detectada y confianza
    """
    # Convertir a HSV para análisis de color
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Definir rangos de color para cada fruta
    # Manzana: predominantemente roja o verde
    # Banana: predominantemente amarilla
    # Naranja: predominantemente naranja
    
    # Máscaras para cada color
    # Rojo (manzana roja)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Verde (manzana verde)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Amarillo (banana)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Naranja (naranja)
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Marrón (banana madura/podrida o manzana podrida)
    lower_brown = np.array([10, 50, 20])
    upper_brown = np.array([30, 200, 150])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Contar píxeles de cada color
    total_pixels = image.shape[0] * image.shape[1]
    
    red_ratio = np.sum(mask_red > 0) / total_pixels
    green_ratio = np.sum(mask_green > 0) / total_pixels
    yellow_ratio = np.sum(mask_yellow > 0) / total_pixels
    orange_ratio = np.sum(mask_orange > 0) / total_pixels
    brown_ratio = np.sum(mask_brown > 0) / total_pixels
    
    # Determinar tipo de fruta
    scores = {
        'Manzana 🍎': red_ratio + green_ratio * 0.8,
        'Banana 🍌': yellow_ratio + brown_ratio * 0.5,
        'Naranja 🍊': orange_ratio
    }
    
    # Normalizar scores
    total_score = sum(scores.values())
    if total_score > 0:
        confidences = {k: v / total_score for k, v in scores.items()}
    else:
        confidences = {k: 0.33 for k in scores.keys()}
    
    # Obtener fruta con mayor score
    detected_fruit = max(scores, key=scores.get)
    confidence = confidences[detected_fruit]
    
    return {
        'fruit_type': detected_fruit,
        'confidence': confidence,
        'all_scores': confidences
    }


def show_model_status():
    """Muestra el estado del modelo en el sidebar."""
    status = check_model_status()
    
    if status['model_loadable']:
        st.sidebar.success("✅ Modelo cargado correctamente")
        with st.sidebar.expander("📋 Info del Modelo", expanded=False):
            st.markdown(f"""
            - **Arquitectura**: MobileNetV2 (PyTorch)
            - **Archivo**: `{Path(status['model_path']).name}`
            - **Dispositivo**: {status['device']}
            - **Clases**: Fresca, Podrida
            - **Entrada**: 224×224 + normalización ImageNet
            """)
            if status['info'] and status['info'].get('val_acc') != 'N/A':
                st.metric("Precisión de Validación", f"{status['info']['val_acc']:.1f}%")
    else:
        st.sidebar.error("❌ Modelo no encontrado")
        st.sidebar.warning(f"Esperado: `models/fruit_quality_baseline.pth`")
        if status['error']:
            st.sidebar.code(status['error'])
    
    return status['model_loadable']


@st.cache_resource
def load_classification_model(model_path: str = None):
    """Carga y cachea el modelo de clasificación."""
    try:
        model, device, info = load_inference_model(model_path)
        return model, device, info
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None, None, None


def predict_quality(model, image: np.ndarray, device: torch.device, preprocess: str = None) -> dict:
    """
    Predice la calidad de la fruta en la imagen.
    
    Returns:
        Diccionario con predicción, confianza y probabilidades
    """
    return predict(model, image, device, preprocess)


def save_evaluation(image, segmented, result, fruit_type, seg_method, output_dir):
    """Guarda los resultados de evaluación en la carpeta streamlit_samples."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{result['prediction'].lower()}_{timestamp}"
    
    # Guardar original
    Image.fromarray(image).save(os.path.join(output_dir, f"{prefix}_original.png"))
    
    # Guardar segmentada
    if segmented is not None:
        Image.fromarray(segmented).save(os.path.join(output_dir, f"{prefix}_segmented.png"))
    
    # Guardar info del resultado
    import json
    result_info = {
        'prediction': result['prediction'],
        'confidence': result['confidence'],
        'probabilities': result['probabilities'],
        'fruit_type': fruit_type,
        'segmentation_method': seg_method,
        'timestamp': timestamp
    }
    with open(os.path.join(output_dir, f"{prefix}_result.json"), 'w', encoding='utf-8') as f:
        json.dump(result_info, f, indent=2, ensure_ascii=False)
    
    return prefix


def main():
    # Header
    st.markdown('<h1 class="main-header">🍎 Analizador de Calidad de Frutas</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - Configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Auto-verificación del modelo
        model_available = show_model_status()
        
        st.markdown("---")
        
        # Configuración de preprocesamiento
        st.subheader("🔍 Preprocesamiento")
        seg_method = st.selectbox(
            "Método",
            ["Ninguno (Original)", "GrabCut", "HSV + Morfología"],
            help="Selecciona el preprocesamiento: Ninguno para inferencia directa, o segmentación para remover el fondo"
        )
        
        # Parámetros de segmentación (solo mostrar si está seleccionado)
        iterations = 5
        margin = 10
        kernel_size = 5
        
        if seg_method == "GrabCut":
            iterations = st.slider("Iteraciones", 1, 10, 5)
            margin = st.slider("Margen", 5, 30, 10)
        elif seg_method == "HSV + Morfología":
            kernel_size = st.slider("Tamaño de Kernel", 3, 11, 5, step=2)
        
        # Panel informativo
        st.markdown("---")
        st.subheader("📋 Info del Proceso")
        st.markdown(f"""
        <div class="info-box">
            <b>Preprocesamiento:</b> {seg_method}<br>
            <b>Modelo:</b> MobileNetV2 (PyTorch)<br>
            <b>Entrada:</b> Redimensionar 224×224, normalización ImageNet<br>
            <b>Clases:</b> Fresca / Podrida
        </div>
        """, unsafe_allow_html=True)
    
    # Contenido principal
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📤 Subir Imagen")
        uploaded_file = st.file_uploader(
            "Elige una imagen de fruta",
            type=['jpg', 'jpeg', 'png'],
            help="Sube una imagen JPG o PNG de una fruta"
        )
        
        if uploaded_file is not None:
            # Cargar imagen
            image = Image.open(uploaded_file).convert('RGB')
            image_np = np.array(image)
            
            st.image(image, caption="Imagen Original", use_container_width=True)
    
    with col2:
        if uploaded_file is not None:
            st.subheader("🔬 Resultados del Análisis")
            
            # Manejar preprocesamiento según selección
            if seg_method == "Ninguno (Original)":
                # Sin segmentación, usar imagen original
                segmented = image_np
                mask = np.ones(image_np.shape[:2], dtype=np.uint8) * 255
                method_info = "Ninguno - usando imagen original"
            else:
                # Aplicar segmentación
                with st.spinner("Aplicando segmentación..."):
                    try:
                        if seg_method == "GrabCut":
                            seg_result = segment_image(
                                image_np, 
                                method="grabcut",
                                iterations=iterations,
                                margin=margin
                            )
                        else:  # HSV + Morfología
                            seg_result = segment_image(
                                image_np, 
                                method="hsv",
                                kernel_size=kernel_size
                            )
                        
                        segmented = seg_result['segmented']
                        mask = seg_result['mask']
                        method_info = seg_result['method_info']
                        
                    except Exception as e:
                        st.error(f"Error en segmentación: {e}")
                        segmented = image_np
                        mask = np.ones(image_np.shape[:2], dtype=np.uint8) * 255
                        method_info = "Respaldo (sin segmentación)"
            
            # Mostrar resultados de segmentación en 3 columnas
            seg_col1, seg_col2, seg_col3 = st.columns(3)
            
            with seg_col1:
                st.image(image_np, caption="Original", use_container_width=True)
            
            with seg_col2:
                # Mostrar máscara
                mask_display = np.stack([mask, mask, mask], axis=2)
                st.image(mask_display, caption="Máscara", use_container_width=True)
            
            with seg_col3:
                st.image(segmented, caption="Procesada", use_container_width=True)
            
            st.info(f"**Preprocesamiento:** {method_info}")
            
            # Botón de evaluación
            st.markdown("---")
            
            if st.button("🔮 Evaluar Calidad", type="primary", use_container_width=True):
                if not model_available:
                    st.error("""
                    ❌ ¡Modelo no disponible!
                    
                    Por favor asegúrate de que el modelo existe en:
                    `models/fruit_quality_baseline.pth`
                    """)
                else:
                    with st.spinner("Analizando calidad de la fruta..."):
                        # Cargar modelo (usando baseline por defecto)
                        model, device, info = load_classification_model()
                        
                        if model is not None:
                            # Obtener predicción de calidad
                            result = predict_quality(model, segmented, device)
                            
                            # Detectar tipo de fruta
                            fruit_info = detect_fruit_type(image_np)
                            
                            # Traducir resultado
                            quality_es = "Fresca" if result['prediction'] == 'Fresh' else "Podrida"
                            
                            # Mostrar resultados
                            st.markdown("### 📊 Resultados de la Predicción")
                            
                            # Tipo de fruta detectada
                            st.markdown(f"""
                            <div class="fruit-type-box">
                                <b>Fruta Identificada:</b> {fruit_info['fruit_type']}<br>
                                <small>Confianza: {fruit_info['confidence']*100:.1f}%</small>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Calidad
                            pred_class = "prediction-fresh" if result['prediction'] == 'Fresh' else "prediction-rotten"
                            emoji = "✅" if result['prediction'] == 'Fresh' else "⚠️"
                            
                            st.markdown(f"""
                            <div class="{pred_class}">
                                {emoji} <b>Calidad: {quality_es}</b><br>
                                Confianza: {result['confidence']*100:.1f}%
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Distribución de probabilidades
                            st.markdown("#### Distribución de Probabilidad")
                            prob_col1, prob_col2 = st.columns(2)
                            
                            with prob_col1:
                                st.metric(
                                    "🍏 Fresca",
                                    f"{result['probabilities']['Fresh']*100:.1f}%"
                                )
                            
                            with prob_col2:
                                st.metric(
                                    "🍂 Podrida",
                                    f"{result['probabilities']['Rotten']*100:.1f}%"
                                )
                            
                            # Barra de progreso
                            st.progress(result['probabilities']['Fresh'])
                            
                            # Mostrar detalles de detección de fruta
                            with st.expander("🔎 Detalles de detección de tipo de fruta"):
                                for fruit, score in fruit_info['all_scores'].items():
                                    st.write(f"{fruit}: {score*100:.1f}%")
                            
                            # Guardar evaluación
                            output_dir = PROJECT_ROOT / "outputs" / "streamlit_samples"
                            saved = save_evaluation(
                                image_np, segmented, result,
                                fruit_info['fruit_type'],
                                seg_method, str(output_dir)
                            )
                            st.success(f"✅ Evaluación guardada: {saved}")
        
        else:
            st.info("👈 Sube una imagen para comenzar el análisis")
            
            # Mostrar instrucciones
            st.markdown("""
            ### Cómo usar:
            1. **Sube** una imagen de fruta (manzana, banana o naranja)
            2. **Selecciona** el método de preprocesamiento en el panel lateral
            3. **Haz clic** en "Evaluar Calidad" para obtener la predicción
            
            ### Opciones de preprocesamiento:
            - **Ninguno (Original)**: Inferencia directa sobre la imagen subida
            - **GrabCut**: Segmentación iterativa basada en grafos
            - **HSV + Morfología**: Segmentación basada en color
            
            ### Frutas soportadas:
            - 🍎 Manzanas (frescas y podridas)
            - 🍌 Bananas (frescas y podridas)  
            - 🍊 Naranjas (frescas y podridas)
            
            ### El sistema detectará:
            - **Tipo de fruta**: Manzana, Banana o Naranja
            - **Calidad**: Fresca o Podrida
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888;">
        Analizador de Calidad de Frutas | Modo Demo | No requiere dataset | 2026
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

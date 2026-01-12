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

from src.segmentation import segment_image, get_edges, get_hsv_channels, apply_filters, get_solidity
from src.inference import load_inference_model, predict, check_model_status, get_device
from src.config import CLASS_NAMES, DEFAULT_MODEL_PATH, SEGMENTATION_METHODS, FRUIT_COLOR_RANGES


# Page configuration
if __name__ == "__main__":
    st.set_page_config(
        page_title="🎓 Laboratorio Académico de Frutas",
        page_icon="🍎",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Custom CSS
st.markdown("""
<style>
    /* Industrial Tech Aesthetic - Charcoal & Amber */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap');

    :root {
        --bg-color: #0d0d0d;
        --card-bg: #161616;
        --border-color: #262626;
        --accent: #f59e0b; /* Amber */
        --accent-hover: #d97706;
        --text-bright: #ffffff;
        --text-subtle: #a3a3a3;
        --danger: #ef4444;
        --success: #10b981;
    }

    /* Hide Streamlit Header */
    header, [data-testid="stHeader"] {
        visibility: hidden;
        height: 0px;
    }

    .stApp {
        background-color: var(--bg-color);
        color: var(--text-bright);
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Sidebar - Deep Neutral & High Contrast Navigation */
    [data-testid="stSidebar"] {
        background-color: #080808;
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] * {
        color: #e5e5e5 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown p {
        color: #d4d4d4 !important;
        font-size: 0.95rem;
    }
    
    /* Navigation Links Fix */
    section[data-testid="stSidebarNav"] li a span {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* Typography */
    .main-header {
        font-size: 4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0px;
        letter-spacing: -2px;
        color: var(--accent);
        text-transform: uppercase;
    }

    .sub-header {
        text-align: center;
        color: var(--text-subtle);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 4px;
        margin-bottom: 4rem;
        font-weight: 300;
    }

    /* Cards - Industrial Style */
    .glass-card {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 4px; /* Industrial sharp edges slightly rounded */
        padding: 24px;
        margin-bottom: 24px;
        transition: border-color 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: #404040;
    }

    .section-title {
        color: var(--accent);
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 20px;
        text-transform: uppercase;
        letter-spacing: 2px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .section-title::after {
        content: "";
        flex-grow: 1;
        height: 1px;
        background: var(--border-color);
    }

    /* AI Input Frame - Danger Amber Border */
    .ai-input-frame {
        border: 1px solid var(--danger);
        background: rgba(239, 68, 68, 0.05);
        padding: 16px;
        position: relative;
        margin-top: 32px;
    }

    .ai-badge {
        background: var(--danger);
        color: black;
        font-size: 10px;
        font-weight: 800;
        padding: 2px 8px;
        position: absolute;
        top: -10px;
        left: 16px;
    }

    /* Results */
    .result-box {
        border-left: 2px solid var(--accent);
        background: #1a1a1a;
        padding: 20px;
    }

    /* Buttons - Industrial Amber */
    .stButton > button {
        background-color: var(--accent) !important;
        color: black !important;
        border-radius: 2px !important;
        border: none !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        padding: 12px !important;
    }

    .stButton > button:hover {
        background-color: var(--accent-hover) !important;
        transform: scale(1.01);
    }

    /* Tabs UI */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid var(--border-color);
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--text-subtle);
        background: transparent !important;
        font-weight: 600;
        font-size: 14px;
    }

    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
    }

    /* Inputs & Selects */
    div[data-baseweb="select"] > div {
        background-color: var(--card-bg) !important;
        border-color: var(--border-color) !important;
        color: white !important;
    }

    /* Spectra Industrial UI - v6 Clean & Central */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        margin-bottom: 20px;
    }

    .info-block {
        background: rgba(245, 158, 11, 0.03);
        border-left: 3px solid var(--accent);
        padding: 18px;
        margin: 20px 0;
        color: #ddd;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .tech-header {
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent);
        font-size: 0.8rem;
        margin-top: 30px;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        border-bottom: 1px solid rgba(245, 158, 11, 0.2);
        padding-bottom: 8px;
    }

    /* AI Final Result Card */
    .verdict-card {
        background: #111;
        border: 1px solid var(--accent);
        padding: 25px;
        border-radius: 8px;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)


def extract_features(image: np.ndarray, mask: np.ndarray) -> dict:
    """
    Extracción de características avanzada. 
    Si la máscara es el cuadro completo, intenta una auto-aislamiento.
    """
    h, w = mask.shape
    mask_coverage = np.sum(mask > 0) / (h * w)
    
    # Si la máscara es casi toda la imagen, intentamos aislar el objeto por saturación
    if mask_coverage > 0.95:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        s_channel = hsv[:, :, 1]
        _, rough_mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Limpiar un poco
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        rough_mask = cv2.morphologyEx(rough_mask, cv2.MORPH_OPEN, kernel)
        mask = rough_mask

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}
    
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    # 1. Relación de Aspecto Rotada
    rect = cv2.minAreaRect(cnt)
    _, (rw, rh), _ = rect
    rotated_ar = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 1.0
    
    # 2. Circularidad y Solidez
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
    solidity = get_solidity(cnt)
    
    # 3. Análisis de Color HSV por Cobertura
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    color_scores = {}
    
    for fruit, ranges in FRUIT_COLOR_RANGES.items():
        fruit_mask = np.zeros(mask.shape, dtype=np.uint8)
        for lower, upper in ranges:
            m = cv2.inRange(hsv, np.array(lower), np.array(upper))
            fruit_mask = cv2.bitwise_or(fruit_mask, m)
        
        # Intersección con el objeto
        fruit_obj_mask = cv2.bitwise_and(fruit_mask, mask)
        # Cobertura relativa al objeto, no a la imagen completa
        obj_pixels = np.sum(mask > 0)
        coverage = np.sum(fruit_obj_mask > 0) / obj_pixels if obj_pixels > 0 else 0
        color_scores[fruit] = coverage
        
    masked_hsv = hsv[mask > 0]
    mean_hsv = np.mean(masked_hsv, axis=0) if len(masked_hsv) > 0 else [0, 0, 0]
        
    return {
        'área': area,
        'perímetro': perimeter,
        'relación_aspecto_rotada': rotated_ar,
        'circularidad': circularity,
        'solidez': solidity,
        'hsv_promedio': mean_hsv,
        'color_coverage': color_scores,
        'refined_mask': mask # Devolvemos la máscara refinada para visualización
    }


def detect_fruit_type_improved(features: dict) -> dict:
    """
    Motor de Reconocimiento V3: Jerarquía de Forma Rotada + Cobertura de Color.
    """
    if not features:
        return {'fruit_type': "No detectado ❓", 'confidence': 0.0, 'all_scores': {}}
        
    ar = features.get('relación_aspecto_rotada', 1.0)
    circ = features.get('circularidad', 0.0)
    sol = features.get('solidez', 0.0)
    color_coverage = features.get('color_coverage', {})
    
    scores = {"Manzana 🍎": 0.0, "Banana 🍌": 0.0, "Naranja 🍊": 0.0}
    
    # --- 1. CRITERIO DE FORMA (Peso Máx: 3.0) ---
    # Banana: Alargada (AR > 2.0)
    if ar > 2.0: scores["Banana 🍌"] += 2.0
    elif ar > 1.7: scores["Banana 🍌"] += 1.0
    if sol < 0.92: scores["Banana 🍌"] += 1.0
    
    # Naranja: Muy circular y sólida
    if circ > 0.8 and ar < 1.25: scores["Naranja 🍊"] += 2.0
    if sol > 0.95: scores["Naranja 🍊"] += 1.0
    
    # Manzana: Circularidad media-alta
    if 0.65 < circ < 0.9: scores["Manzana 🍎"] += 1.5
    if 0.85 < sol < 0.98: scores["Manzana 🍎"] += 1.0
    
    # Penalización Crítica: Si es Redonda (~1) NO puede ser Banana
    if ar < 1.4 and circ > 0.65:
        scores["Banana 🍌"] -= 5.0

    # --- 2. CRITERIO DE COLOR (Peso Máx: 3.0) ---
    for fruit in scores:
        coverage = color_coverage.get(fruit, 0.0)
        scores[fruit] += (coverage * 3.0)
        
    # Decisión final
    detected_fruit = max(scores, key=scores.get)
    max_score = scores[detected_fruit]
    
    # Si el puntaje es muy bajo, no estamos seguros
    if max_score < 1.0:
        return {'fruit_type': "No detectado ❓", 'confidence': 0.0, 'all_scores': scores}

    total_score = sum(abs(v) for v in scores.values())
    confidence = (max_score / total_score) if total_score > 0 else 0.33
    
    return {
        'fruit_type': detected_fruit,
        'confidence': confidence,
        'all_scores': scores
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
    st.markdown('<div class="main-header">SPECTRA</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated Fruit Quality Analysis Engine</div>', unsafe_allow_html=True)
    
    # Sidebar - Configuración
    with st.sidebar:
        st.markdown('<div class="section-title">CONFIGURATION</div>', unsafe_allow_html=True)
        
        # Auto-verificación del modelo
        model_available = show_model_status()
        
        st.markdown("---")
        
        # Configuración de preprocesamiento
        st.subheader("🔍 Segmentación (Preprocesamiento)")
        seg_method = st.selectbox(
            "Método de Segmentación",
            ["HSV + Morfología", "GrabCut", "Ninguno (Original)"],
            index=0,
            help="Selecciona el método para la segmentación completa del objeto"
        )
        
        # Parámetros de segmentación (solo mostrar si está seleccionado)
        iterations = 5
        margin = 10
        kernel_size = 5
        
        if seg_method == "GrabCut":
            iterations = st.slider("Iteraciones GrabCut", 1, 10, 5)
            margin = st.slider("Margen inicial", 5, 30, 10)
        elif seg_method == "HSV + Morfología":
            kernel_size = st.slider("Tamaño Kernel (Morfología)", 3, 11, 5, step=2)
        
        # Panel académico
        st.markdown("---")
        st.subheader("🎓 Explicación Súper Simple")
        with st.expander("🤔 ¿Cómo funciona esto?", expanded=True):
            st.markdown("""
            **1. Segmentación Parcial (El Boceto)**
            Es como si estuviéramos dibujando los bordes de la fruta o marcando dónde hay manchas del mismo color. No sacamos la fruta de la foto, solo "señalamos" dónde está.
            
            **2. Segmentación Completa (El Sticker)**
            Es como si recortáramos la fruta con unas tijeras para convertirla en un **sticker**. Quitamos todo lo que no sirve (el fondo blanco o la mesa) y nos quedamos solo con la fruta limpiecita para poder analizarla.
            
            **¿Cómo logramos ese corte perfecto?**
            - Primero buscamos colores parecidos.
            - Luego rellenamos los huequitos que falten.
            - Al final, borramos los puntitos de basura que queden alrededor.
            """)
    
    # Flujo de Carga de Archivo (Arriba y central)
    st.markdown('<div class="tech-header">📥 ENTRADA DE IMAGEN</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Carga la imagen de la fruta aquí",
        type=['jpg', 'jpeg', 'png', 'webp']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        
        st.divider()
        st.subheader("🔬 Laboratorio de Análisis y Clasificación")
        
        # Aplicar segmentación completa (Cálculo interno)
        with st.spinner("Procesando..."):
            if seg_method == "Ninguno (Original)":
                segmented = image_np
                mask = np.ones(image_np.shape[:2], dtype=np.uint8) * 255
                method_info = "Sin segmentación"
            else:
                try:
                    if seg_method == "GrabCut":
                        seg_result = segment_image(image_np, method="grabcut", iterations=iterations, margin=margin)
                    else:
                        seg_result = segment_image(image_np, method="hsv", kernel_size=kernel_size)
                    
                    segmented = seg_result['segmented']
                    mask = seg_result['mask']
                    method_info = seg_result['method_info']
                except Exception as e:
                    st.error(f"Error: {e}")
                    segmented = image_np
                    mask = np.ones(image_np.shape[:2], dtype=np.uint8) * 255
                    method_info = "Fallback"

        # Extraer descriptores y obtener máscara refinada académica
        features = extract_features(image_np, mask)
        viz_mask = features.get('refined_mask', mask) if features else mask

        if seg_method == "HSV + Morfología":
            # --- PIPELINE HSV ---
            hsv_tabs = st.tabs(["1. 🧪 Canales Color", "2. 🧬 Máscaras", "3. 🧼 Morfología", "4. 📊 Datos Finales"])
            
            with hsv_tabs[0]:
                st.markdown('<div class="tech-header">ANÁLISIS DE CANALES (HSV)</div>', unsafe_allow_html=True)
                st.markdown('<div class="info-block">Descomponemos la luz: <b>H</b> (Tono), <b>S</b> (Saturación), <b>V</b> (Brillo). Observa cómo el canal S suele resaltar mejor la fruta.</div>', unsafe_allow_html=True)
                channels = get_hsv_channels(image_np)
                c1, c2, c3 = st.columns(3)
                c1.image(channels['H'], caption="H: Qué color es", use_container_width=True)
                c2.image(channels['S'], caption="S: Qué tan puro es", use_container_width=True)
                c3.image(channels['V'], caption="V: Cuánta luz tiene", use_container_width=True)

            with hsv_tabs[1]:
                st.markdown('<div class="tech-header">GENERACIÓN DE MÁSCARAS</div>', unsafe_allow_html=True)
                st.markdown('<div class="info-block">Creamos dos "moldes": uno basado en la forma (Otsu) y otro en el color específico de la fruta.</div>', unsafe_allow_html=True)
                
                # Simulación visual de pasos HSV
                hsv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
                # Otsu
                _, otsu_m = cv2.threshold(hsv_img[:,:,1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # Color (aprox)
                # Usamos el resultado 'segmented' para inferir qué pasó, o calculamos rápido
                # Para fines educativos visualizamos lo que tenemos
                c1, c2 = st.columns(2)
                c1.image(otsu_m, caption="Máscara de Forma (Otsu en S)", use_container_width=True)
                c2.image(seg_result.get('mask', mask), caption="Máscara Combinada Final", use_container_width=True)

            with hsv_tabs[2]:
                st.markdown('<div class="tech-header">REFINAMIENTO (MORFOLOGÍA)</div>', unsafe_allow_html=True)
                st.markdown('<div class="info-block">Usamos operaciones matemáticas (Erosión/Dilatación) para limpiar ruido (puntitos blancos) y cerrar huecos negros dentro de la fruta.</div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.image(seg_result.get('mask', mask), caption="Máscara Limpia", use_container_width=True)
                c2.image(seg_result.get('segmented', segmented), caption="Fruta Recortada", use_container_width=True)

            with hsv_tabs[3]:
                st.markdown('<div class="tech-header">RESULTADOS ANALÍTICOS</div>', unsafe_allow_html=True)
                if features:
                    c1, c2 = st.columns([1,1])
                    with c1:
                        st.image(seg_result.get('segmented', segmented), caption="Final para Clasificación", width=300)
                    with c2:
                         st.markdown("#### Métricas")
                         st.write(f"- **Área**: {features['área']:.0f} px")
                         st.write(f"- **Circularidad**: {features['circularidad']:.2f}")

        elif seg_method == "GrabCut":
            # --- PIPELINE GRABCUT ---
            gc_tabs = st.tabs(["1. 📦 Inicialización", "2. ⚙️ Proceso Iterativo", "3. 🍎 Resultado", "4. 📊 Datos"])
            
            with gc_tabs[0]:
                st.markdown('<div class="tech-header">DEFINICIÓN DE REGIÓN (ROI)</div>', unsafe_allow_html=True)
                st.markdown('<div class="info-block">GrabCut necesita una pista inicial: un rectángulo donde <b>probablemente</b> está la fruta. Todo lo de afuera es 100% fondo.</div>', unsafe_allow_html=True)
                
                # Dibujar rectángulo simulado
                viz_rect = image_np.copy()
                h, w = viz_rect.shape[:2]
                # Default logic from segmentation.py
                m = margin
                cv2.rectangle(viz_rect, (m, m), (w-2*m, h-2*m), (255, 0, 0), 3)
                st.image(viz_rect, caption=f"Rectángulo Inicial (Margen {margin}px)", use_container_width=True)

            with gc_tabs[1]:
                st.markdown('<div class="tech-header">MODELADO GAUSSIANO (GMM)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-block">El algoritmo ejecuta <b>{iterations} iteraciones</b>. En cada paso, construye un modelo de color para el "Fondo" y otro para el "Frente", refinando el borde píxel a píxel.</div>', unsafe_allow_html=True)
                st.image(seg_result.get('mask', mask), caption="Máscara de Probabilidad Resultante", use_container_width=True)

            with gc_tabs[2]:
                st.markdown('<div class="tech-header">EXTRACCIÓN FINAL</div>', unsafe_allow_html=True)
                st.image(seg_result.get('segmented', segmented), caption="Objeto Aislado", use_container_width=True)
                
            with gc_tabs[3]:
                 st.markdown('<div class="tech-header">ANÁLISIS GEOMÉTRICO</div>', unsafe_allow_html=True)
                 if features:
                     st.dataframe(features)

        else:
            st.image(image_np, caption="Imagen Original (Sin Segmentación)")

        # Sección común de Descriptores detallados (siempre útil)
        st.markdown("---")
        st.markdown('<div class="tech-header">🔬 REPORTE DETALLADO DE FORMA Y COLOR</div>', unsafe_allow_html=True)
        if features:
             c_tab1, c_tab2 = st.columns([1.5, 1])
             with c_tab1:
                 data = {
                     "Atributo": ["📏 Área", "📉 Perímetro", "🖼️ Aspecto", "⭕ Circularidad", "💎 Solidez"],
                     "Valor": [
                         f"{features['área']:.0f} px²",
                         f"{features['perímetro']:.1f} px",
                         f"{features['relación_aspecto_rotada']:.2f}",
                         f"{features['circularidad']:.4f}",
                         f"{features['solidez']:.4f}"
                     ]
                 }
                 st.table(data)
             with c_tab2:
                 st.markdown("##### Cobertura Color")
                 for fruit, coverage in features['color_coverage'].items():
                     val = coverage * 100
                     color = "#10b981" if val > 30 else "#f59e0b" if val > 10 else "#6b7280"
                     st.markdown(f"<span style='color:{color}; font-weight:bold;'>{fruit}: {val:.1f}%</span>", unsafe_allow_html=True)

        # RECONOCEDOR
        st.markdown("---")
        if st.button("🚀 Ejecutar Reconocedor (Clasificador)", type="primary", use_container_width=True):
            if model_available:
                with st.spinner("Analizando calidad y forma..."):
                    # Inferencia de Deep Learning (Calidad)
                    model, device, info = load_classification_model()
                    dl_result = predict_quality(model, segmented, device)
                    
                    # Análisis de Forma (Tipo de Fruta)
                    fruit_info = detect_fruit_type_improved(features)
                    
                    # Mostrar Resultado Final
                    st.markdown('<div class="verdict-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">🏆 Veredicto de Spectra</div>', unsafe_allow_html=True)
                    
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #6366f1, #a855f7); padding: 1.5rem; border-radius: 1rem; text-align: center;">
                            <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.8;">Identificación</div>
                            <div style="font-size: 2rem; font-weight: 700; margin: 0.5rem 0;">{fruit_info['fruit_type']}</div>
                            <div style="font-size: 0.9rem; opacity: 0.7;">Análisis de Forma y Geometría</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with res_col2:
                        quality_es = "Fresca" if dl_result['prediction'] == 'Fresh' else "Dañada"
                        color_grad = "linear-gradient(135deg, #059669, #10b981)" if dl_result['prediction'] == 'Fresh' else "linear-gradient(135deg, #dc2626, #ef4444)"
                        st.markdown(f"""
                        <div style="background: {color_grad}; padding: 1.5rem; border-radius: 1rem; text-align: center;">
                            <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.8;">Estado de Calidad</div>
                            <div style="font-size: 2rem; font-weight: 700; margin: 0.5rem 0;">{quality_es}</div>
                            <div style="font-size: 0.9rem; opacity: 0.7;">Confianza IA: {dl_result['confidence']*100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Explicación del Motor de Decisión
                    with st.expander("🎓 ¿Cómo tomó la decisión el sistema?", expanded=False):
                        st.markdown("""
                        El 'cerebro' del programa usa un **Motor de Puntuación Híbrido**:
                        1. **Filtro de Forma:** Primero revisa si la forma es 'larga' o 'redonda'. Si es muy redonda, le prohíbe ser banana.
                        2. **Votación por Color:** Luego suma puntos si el color de la fruta coincide con los patrones guardados (Naranja, Rojo, Amarillo).
                        3. **Consenso:** La fruta con más puntos gana. La confianza (%) indica qué tan lejos estuvo el primer lugar del segundo.
                        
                        *Este método es lo que en IA llamamos un 'Sistemas Basado en Reglas y Lógica Borrosa (Fuzzy Logic)'.*
                        """)

                    # Tabla de probabilidades de calidad
                    with st.expander("📊 Detalles del Sistema de Puntuación (Reconocedor)"):
                        st.write("El reconocedor asigna puntos basados en descriptores de forma y cobertura de color:")
                        
                        st.markdown(f"""
                        **Análisis Geométrico:**
                        - Rel. Aspecto (Rotada): `{features['relación_aspecto_rotada']:.2f}` (Banana > 1.6)
                        - Circularidad: `{features['circularidad']:.2f}` (Naranja > 0.8)
                        - Solidez: `{features['solidez']:.2f}`
                        """)
                        
                        for fruit, score in fruit_info['all_scores'].items():
                            st.progress(max(0, min(score/6.0, 1.0)), text=f"{fruit}: {score:.1f} pts")
                        st.info("Un puntaje alto en forma alargada favorece a la Banana. La cobertura de color se mide comparando los píxeles con el rango patrón HSV de cada fruta.")

                    with st.expander("🧠 Probabilidades del Modelo Neuronal (Calidad)"):
                        st.json(dl_result['probabilities'])
                    
                    # Guardar
                    output_dir = PROJECT_ROOT / "outputs" / "streamlit_samples"
                    save_evaluation(image_np, segmented, dl_result, fruit_info['fruit_type'], seg_method, str(output_dir))
            else:
                st.error("Modelo no cargado. Verifica 'models/fruit_quality_baseline.pth'")
                
                st.markdown(f"<p style='text-align:center; color:var(--text-dim);'>Imagen procesada lista para el modelo <b>MobileNetV2</b></p>", unsafe_allow_html=True)
            
    else:
            st.info("👈 Esperando imagen para iniciar el proceso académico...")
            st.markdown("""
            ### Flujo Académico Implementado:
            1. **Segmentación Parcial**:
               - **Discontinuidad**: Usamos Canny para encontrar bordes.
               - **Similaridad**: Usamos máquinas de color (HSV) para agrupar píxeles.
            2. **Segmentación Completa**: Algoritmo GrabCut o HSV con morfología para aislar el objeto.
            3. **Extracción de Características**:
               - Calculamos descriptores de **Forma** (Área, Perímetro, Aspecto, Circularidad).
               - Extraemos descriptores de **Color** (Promedio RGB).
            4. **Descripción**: Generamos un vector de características del objeto.
            5. **Reconocedor**: Un sistema híbrido (Heurísticas de forma + Deep Learning) toma la decisión final.
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.8rem;">
        Image Analysis 2026 | Práctica de Segmentación y Reconocimiento Académico
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

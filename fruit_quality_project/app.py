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
st.set_page_config(
    page_title="🎓 Laboratorio Académico de Frutas",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a0b2e 0%, #2d1b4e 50%, #16213e 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0324 0%, #1a0b2e 100%);
    }
    .stMarkdown, .stText, p, h1, h2, h3, label {
        color: #ede9fe !important;
    }
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
    .academic-box {
        background-color: rgba(107, 70, 193, 0.1);
        border: 1px solid #9f7aea;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
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
    .step-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
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
    elif ar > 1.5: scores["Banana 🍌"] += 1.0
    if sol < 0.92: scores["Banana 🍌"] += 1.0
    
    # Naranja: Muy circular y sólida
    if circ > 0.8 and ar < 1.25: scores["Naranja 🍊"] += 2.0
    if sol > 0.95: scores["Naranja 🍊"] += 1.0
    
    # Manzana: Circularidad media-alta
    if 0.65 < circ < 0.9: scores["Manzana 🍎"] += 1.5
    if 0.85 < sol < 0.98: scores["Manzana 🍎"] += 1.0

    # --- 2. CRITERIO DE COLOR (Peso Máx: 3.0) ---
    for fruit in scores:
        coverage = color_coverage.get(fruit, 0.0)
        scores[fruit] += (coverage * 3.0)
        
    # Penalización Crítica: Si es Redonda (~1) NO puede ser Banana
    if ar < 1.4 and circ > 0.65:
        scores["Banana 🍌"] -= 5.0

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
    st.markdown('<h1 class="main-header">🍎 Analizador Académico de Frutas</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - Configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
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
    
    # Contenido principal
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📤 Entrada")
        uploaded_file = st.file_uploader(
            "Subir imagen de fruta",
            type=['jpg', 'jpeg', 'png', 'webp']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            image_np = np.array(image)
            st.image(image, caption="Original RGB", use_container_width=True)
            
            # 1. Segmentación Parcial: Discontinuidad (Bordes)
            st.markdown("---")
            st.subheader("📉 Seg. Parcial: Discontinuidad")
            edges = get_edges(image_np)
            st.image(edges, caption="Detección de Bordes (Canny)", use_container_width=True)
    
    with col2:
        if uploaded_file is not None:
            st.subheader("🔬 Proceso de Segmentación y Descripción")
            
            # Aplicar segmentación completa
            with st.spinner("Procesando segmentación..."):
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

            # Visualización por fases
            tab1, tab2, tab3, tab4 = st.tabs(["🧩 Seg. Parcial", "🛡️ Filtros y Canales", "🎯 Seg. Completa", "📊 Descripción"])
            
            with tab1:
                st.markdown("#### 🧩 Fase 1: Segmentación Parcial")
                with st.expander("📖 Teoría de esta fase", expanded=False):
                    st.markdown("""
                    La segmentación parcial divide la imagen en regiones prometedoras sin aislar el objeto final.
                    - **Similaridad (Color):** Puntos con tonalidades parecidas tienden a pertenecer al mismo objeto. Aquí agrupamos píxeles rojos/naranjas/amarillos.
                    - **Discontinuidad (Bordes):** Detectamos saltos bruscos en el brillo de los píxeles (Algoritmo de Canny). Esto define el 'dibujo' o contorno de la fruta.
                    """)
                col_sim1, col_sim2 = st.columns(2)
                with col_sim1:
                    st.image(viz_mask, caption="Similaridad (Máscara Extraída)", use_container_width=True)
                with col_sim2:
                    st.image(get_edges(image_np), caption="Discontinuidad (Bordes Canny)", use_container_width=True)
            
            with tab2:
                st.markdown("#### 🛡️ Fase 2: Procesamiento de Canales y Filtrado")
                with st.expander("📖 ¿Qué estamos viendo aquí?", expanded=False):
                    st.markdown("""
                    Para que la computadora 'vea' mejor, descomponemos la imagen:
                    - **Canal Hue (Tono):** El color puro. Ayuda a ignorar sombras.
                    - **Canal Saturation (Saturación):** Qué tan intenso es el color. Muy útil para separar la fruta (colorida) del fondo (grisáceo o blanco).
                    - **Canal Value (Brillo):** Intensidad lumínica.
                    - **Filtro Gaussiano:** Suavizamos la imagen para que el ruido (granulado) no genere errores en el recorte final.
                    """)
                channels = get_hsv_channels(image_np)
                c1, c2, c3 = st.columns(3)
                c1.image(channels['H'], caption="Canal Hue", use_container_width=True)
                c2.image(channels['S'], caption="Canal Saturation", use_container_width=True)
                c3.image(channels['V'], caption="Canal Value", use_container_width=True)
                
                filtered = apply_filters(image_np, method="gaussian", kernel_size=7)
                st.image(filtered, caption="Filtro Gaussiano (Reducción de Ruido)", use_container_width=True)

            with tab3:
                st.markdown("#### 🎯 Fase 3: Segmentación Completa (Aislamiento)")
                with st.expander("📖 El proceso de 'Recorte' Inteligente", expanded=False):
                    st.markdown("""
                    Aquí es donde 'extraemos' la fruta:
                    1. **Umbralización de Otsu:** Un algoritmo calcula automáticamente el punto exacto para separar la fruta del fondo usando la saturación.
                    2. **Morfología Matemática:** Usamos operaciones de 'Cierre' y 'Apertura' para rellenar huecos dentro de la fruta y borrar puntos sueltos en el fondo.
                    3. **Selección de Contorno:** Nos quedamos solo con la forma más grande detectada.
                    """)
                inner_col1, inner_col2 = st.columns(2)
                with inner_col1:
                    st.image(segmented, caption="Segmentación Seleccionada", use_container_width=True)
                with inner_col2:
                    st.image(viz_mask, caption="Objeto Detectado (Máscara Académica)", use_container_width=True)
                st.success(f"Método aplicado: {seg_method}")
            
            with tab4:
                st.markdown("#### 📊 Fase 4: Extracción de Características (Descriptores)")
                with st.expander("📖 ¿Cómo describimos una fruta con números?", expanded=False):
                    st.markdown("""
                    Convertimos la forma en datos estadísticos:
                    - **Relación de Aspecto:** ¿Es más larga que ancha? (Clave para bananas).
                    - **Circularidad:** Qué tanto se parece a un círculo perfecto (0 a 1).
                    - **Solidez:** Qué tan 'perfilada' está la forma. Si tiene muchas hendiduras, la solidez baja.
                    - **Cobertura de Color:** Porcentaje de la fruta que coincide con los colores patrón (ej: 90% naranja).
                    """)
                if features:
                    df_features = {
                        "Descriptor": ["Área (px²)", "Perímetro (px)", "Rel. Aspecto (Rotada)", "Circularidad", "Solidez"],
                        "Valor": [
                            f"{features['área']:.2f}",
                            f"{features['perímetro']:.2f}",
                            f"{features['relación_aspecto_rotada']:.2f}",
                            f"{features['circularidad']:.4f}",
                            f"{features['solidez']:.4f}"
                        ]
                    }
                    st.table(df_features)
                    
                    st.markdown(f"**Análisis de Cobertura de Color:**")
                    for fruit, coverage in features['color_coverage'].items():
                        st.write(f"- {fruit}: {coverage*100:.1f}% de los píxeles coinciden con el rango patrón.")

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
                        st.markdown("### 🏆 Decisión del Reconocedor")
                        
                        res_col1, res_col2 = st.columns(2)
                        
                        with res_col1:
                            st.markdown(f"""
                            <div class="fruit-type-box">
                                <b>Fruta Identificada:</b><br>
                                <span style="font-size: 1.8rem;">{fruit_info['fruit_type']}</span><br>
                                <small>Basado en Forma y Color</small>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with res_col2:
                            quality_es = "Fresca" if dl_result['prediction'] == 'Fresh' else "Podrida"
                            pred_class = "prediction-fresh" if dl_result['prediction'] == 'Fresh' else "prediction-rotten"
                            emoji = "✅" if dl_result['prediction'] == 'Fresh' else "⚠️"
                            
                            st.markdown(f"""
                            <div class="{pred_class}">
                                {emoji} <b>Calidad: {quality_es}</b><br>
                                <small>Confianza Modelo: {dl_result['confidence']*100:.1f}%</small>
                            </div>
                            """, unsafe_allow_html=True)
                        
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
        else:
            st.info("👈 Esperando imagen para iniciar el proceso académico...")
            st.markdown("""
            ### Flujo Académico Implementado:
            1. **Segmentación Parcial**:
               - **Discontinuidad**: Usamos Canny para encontrar bordes.
               - **Similaridad**: Usamos máscaras de color (HSV) para agrupar píxeles.
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

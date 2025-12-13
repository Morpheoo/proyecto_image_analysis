# -*- coding: utf-8 -*-
"""
Práctica 2: Operaciones Aritméticas, Lógicas, Relacionales y Componentes Conexas
Adaptado a Streamlit
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ========= Configuración de página =========
st.set_page_config(
    page_title="Operaciones sobre Imágenes",
    page_icon="📊",
    layout="wide"
)

# Estilo morado
st.markdown("""<style>
.stApp { background: linear-gradient(135deg, #1a0b2e 0%, #2d1b4e 50%, #16213e 100%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0324 0%, #1a0b2e 100%); }
.stMarkdown, .stText, p, h1, h2, h3, label { color: #ede9fe !important; }
.stButton > button { background: linear-gradient(90deg, #6b46c1 0%, #9f7aea 100%); color: white; }
</style>""", unsafe_allow_html=True)

# ========= Funciones auxiliares =========
def ensure_same_size(a, b):
    """Ajusta B al tamaño de A para operar píxel a píxel."""
    if a is None or b is None:
        return a, b
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    if (ha, wa) != (hb, wb):
        b = cv2.resize(b, (wa, ha), interpolation=cv2.INTER_AREA)
    return a, b

def to_gray(img):
    """Convierte a escala de grises."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def binarize(img, thresh):
    """Umbral binario 0/255."""
    gray = to_gray(img)
    _, th = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY)
    return th

def is_binary_image(img):
    """Detecta si una imagen es binaria (solo valores 0/255)."""
    if img is None:
        return False
    gray = to_gray(img)
    unique = np.unique(gray)
    return np.all(np.isin(unique, [0, 255])) and unique.size <= 2

def load_image(uploaded_file):
    """Carga imagen desde archivo subido."""
    if uploaded_file is None:
        return None
    pil_image = Image.open(uploaded_file).convert("RGB")
    return np.array(pil_image)

# ========= Interfaz de Streamlit =========
st.title("Operaciones sobre Imágenes")
st.markdown("### Aritméticas, Lógicas, Relacionales y Componentes Conexas")
st.markdown("---")

# ========= Sidebar para controles =========
with st.sidebar:
    st.header("Configuración")
    
    # Upload de imágenes
    st.subheader("Cargar Imágenes")
    uploaded_A = st.file_uploader("Imagen A", type=["jpg", "jpeg", "png", "bmp", "tiff"], key="img_a")
    uploaded_B = st.file_uploader("Imagen B", type=["jpg", "jpeg", "png", "bmp", "tiff"], key="img_b")
    
    
    st.markdown("---")
    
    # Parámetros
    st.subheader("Parámetros")
    umbral = st.slider("Umbral de binarización", 0, 255, 127)
    escalar = st.slider("Escalar (+/−)", 0, 255, 50)
    factor = st.slider("Factor (×)", 0.1, 3.0, 1.2, 0.1)
    
    st.markdown("---")
    st.subheader("Componentes Conexas")
    cc_source = st.radio("Fuente", ["A", "B", "Resultado"])
    cc_conn = st.radio("Conectividad", [4, 8], index=1)

# Cargar imágenes en session_state solo si cambia el archivo
# Guardar el ID del archivo para detectar cambios
if uploaded_A is not None:
    file_id_a = f"{uploaded_A.name}_{uploaded_A.size}"
    if 'file_id_a' not in st.session_state or st.session_state.file_id_a != file_id_a:
        st.session_state.imgA = load_image(uploaded_A)
        st.session_state.file_id_a = file_id_a
        
if uploaded_B is not None:
    file_id_b = f"{uploaded_B.name}_{uploaded_B.size}"
    if 'file_id_b' not in st.session_state or st.session_state.file_id_b != file_id_b:
        st.session_state.imgB = load_image(uploaded_B)
        st.session_state.file_id_b = file_id_b

imgA = st.session_state.get('imgA', None)
imgB = st.session_state.get('imgB', None)

# Inicializar resultado
if 'result' not in st.session_state:
    st.session_state.result = None

# ========= Tabs para operaciones =========
tab1, tab2, tab3, tab4 = st.tabs([
    "Aritméticas",
    "Lógicas",
    "Relacionales",
    "Componentes Conexas"
])

# ========= TAB 1: OPERACIONES ARITMÉTICAS =========
with tab1:
    st.subheader("Operaciones Aritméticas")
    
    with st.expander("¿Qué son las operaciones aritméticas?", expanded=False):
        st.markdown("""
        Las operaciones aritméticas manipulan los valores de píxeles usando matemática básica:
        
        - **Suma (+)**: Aumenta el brillo sumando un valor constante o combinando dos imágenes
        - **Resta (−)**: Reduce el brillo o calcula diferencias entre imágenes
        - **Multiplicación (×)**: Ajusta el contraste multiplicando por un factor
        - **Lightest (max)**: Toma el valor más alto entre dos imágenes píxel por píxel
        - **Darkest (min)**: Toma el valor más bajo entre dos imágenes píxel por píxel
        
        **Uso típico**: Ajuste de brillo/contraste, mezcla de imágenes, corrección de iluminación.
        """)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Destino:**")
        destino = st.radio("Aplicar operación a:", ["A", "B", "A & B"], key="arith_dest", label_visibility="collapsed")
    
    with col2:
        st.markdown("**Operaciones disponibles:**")
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        
        with col_a:
            if st.button("Suma", use_container_width=True):
                if destino in ["A", "B"]:
                    img = imgA if destino == "A" else imgB
                    if img is not None:
                        if not is_binary_image(img):
                            st.session_state.result = cv2.add(img, np.full_like(img, escalar))
                            st.session_state.result_name = f"Suma +{escalar} a {destino}"
                        else:
                            st.warning(f"Imagen {destino} es binaria. No se aplica escalar.")
                    else:
                        st.error(f"Carga la imagen {destino} primero")
                else:  # A & B
                    if imgA is not None and imgB is not None:
                        a, b = ensure_same_size(imgA, imgB)
                        st.session_state.result = cv2.add(a, b)
                        st.session_state.result_name = "Suma A + B"
                    else:
                        st.error("Carga ambas imágenes A y B")
        
        with col_b:
            if st.button("Resta", use_container_width=True):
                if destino in ["A", "B"]:
                    img = imgA if destino == "A" else imgB
                    if img is not None:
                        if not is_binary_image(img):
                            st.session_state.result = cv2.subtract(img, np.full_like(img, escalar))
                            st.session_state.result_name = f"Resta −{escalar} a {destino}"
                        else:
                            st.warning(f"Imagen {destino} es binaria. No se aplica escalar.")
                    else:
                        st.error(f"Carga la imagen {destino} primero")
                else:  # A & B
                    if imgA is not None and imgB is not None:
                        a, b = ensure_same_size(imgA, imgB)
                        st.session_state.result = cv2.subtract(a, b)
                        st.session_state.result_name = "Resta A − B"
                    else:
                        st.error("Carga ambas imágenes A y B")
        
        with col_c:
            if st.button("Multiplicación", use_container_width=True):
                if destino in ["A", "B"]:
                    img = imgA if destino == "A" else imgB
                    if img is not None:
                        if not is_binary_image(img):
                            st.session_state.result = cv2.convertScaleAbs(img.astype(np.float32) * factor)
                            st.session_state.result_name = f"Mul ×{factor} a {destino}"
                        else:
                            st.warning(f"Imagen {destino} es binaria. No se aplica factor.")
                    else:
                        st.error(f"Carga la imagen {destino} primero")
                else:  # A & B
                    if imgA is not None and imgB is not None:
                        a, b = ensure_same_size(imgA, imgB)
                        prod = (a.astype(np.float32) / 255.0) * (b.astype(np.float32) / 255.0)
                        st.session_state.result = cv2.convertScaleAbs(prod * 255.0)
                        st.session_state.result_name = "Multiplicación A × B"
                    else:
                        st.error("Carga ambas imágenes A y B")
        
        with col_d:
            if st.button("Lightest", use_container_width=True):
                if imgA is not None and imgB is not None:
                    a, b = ensure_same_size(imgA, imgB)
                    st.session_state.result = cv2.max(a, b)
                    st.session_state.result_name = "Lightest (max)"
                else:
                    st.error("Carga ambas imágenes A y B")
        
        with col_e:
            if st.button("Darkest", use_container_width=True):
                if imgA is not None and imgB is not None:
                    a, b = ensure_same_size(imgA, imgB)
                    st.session_state.result = cv2.min(a, b)
                    st.session_state.result_name = "Darkest (min)"
                else:
                    st.error("Carga ambas imágenes A y B")

# ========= TAB 2: OPERACIONES LÓGICAS =========
with tab2:
    st.subheader("Operaciones Lógicas")
    
    with st.expander("¿Qué son las operaciones lógicas?", expanded=False):
        st.markdown("""
        Las operaciones lógicas trabajan con imágenes **binarizadas** (solo valores 0 o 255):
        
        - **AND**: Solo píxeles blancos en AMBAS imágenes permanecen blancos (intersección)
        - **OR**: Píxeles blancos en CUALQUIERA de las imágenes permanecen blancos (unión)
        - **XOR**: Píxeles blancos solo donde difieren las imágenes (diferencia simétrica)
        - **NOT**: Invierte los colores (blanco → negro, negro → blanco)
        
        **Uso típico**: Segmentación, extracción de regiones, máscaras, análisis de formas.
        """)
    
    st.info("Las operaciones lógicas se aplican sobre imágenes binarizadas")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("AND", use_container_width=True):
            if imgA is not None and imgB is not None:
                a, b = ensure_same_size(imgA, imgB)
                binA, binB = binarize(a, umbral), binarize(b, umbral)
                st.session_state.result = cv2.bitwise_and(binA, binB)
                st.session_state.result_name = "AND lógico"
            else:
                st.error("Carga ambas imágenes")
    
    with col2:
        if st.button("OR", use_container_width=True):
            if imgA is not None and imgB is not None:
                a, b = ensure_same_size(imgA, imgB)
                binA, binB = binarize(a, umbral), binarize(b, umbral)
                st.session_state.result = cv2.bitwise_or(binA, binB)
                st.session_state.result_name = "OR lógico"
            else:
                st.error("Carga ambas imágenes")
    
    with col3:
        if st.button("XOR", use_container_width=True):
            if imgA is not None and imgB is not None:
                a, b = ensure_same_size(imgA, imgB)
                binA, binB = binarize(a, umbral), binarize(b, umbral)
                st.session_state.result = cv2.bitwise_xor(binA, binB)
                st.session_state.result_name = "XOR lógico"
            else:
                st.error("Carga ambas imágenes")
    
    with col4:
        if st.button("NOT(A)", use_container_width=True):
            if imgA is not None:
                st.session_state.result = cv2.bitwise_not(imgA)
                st.session_state.result_name = "NOT de A"
            else:
                st.error("Carga la imagen A")
    
    with col5:
        if st.button("NOT(B)", use_container_width=True):
            if imgB is not None:
                st.session_state.result = cv2.bitwise_not(imgB)
                st.session_state.result_name = "NOT de B"
            else:
                st.error("Carga la imagen B")

# ========= TAB 3: OPERACIONES RELACIONALES =========
with tab3:
    st.subheader("Operaciones Relacionales")
    
    with st.expander("¿Qué son las operaciones relacionales?", expanded=False):
        st.markdown("""
        Las operaciones relacionales **comparan** dos imágenes píxel por píxel:
        
        - **A > B**: Píxeles donde A es mayor que B se marcan como blancos
        - **A < B**: Píxeles donde A es menor que B se marcan como blancos
        - **A == B**: Píxeles donde A es igual a B se marcan como blancos
        
        El resultado es una **imagen binaria** que muestra dónde se cumple la condición.
        
        **Uso típico**: Detección de cambios, comparación de imágenes, análisis de diferencias.
        """)
    
    st.info("Compara dos imágenes binarizadas píxel a píxel")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("A > B", use_container_width=True):
            if imgA is not None and imgB is not None:
                binA, binB = binarize(imgA, umbral), binarize(imgB, umbral)
                binA, binB = ensure_same_size(binA, binB)
                st.session_state.result = cv2.compare(binA, binB, cv2.CMP_GT)
                st.session_state.result_name = "A > B"
            else:
                st.error("Carga ambas imágenes")
    
    with col2:
        if st.button("A < B", use_container_width=True):
            if imgA is not None and imgB is not None:
                binA, binB = binarize(imgA, umbral), binarize(imgB, umbral)
                binA, binB = ensure_same_size(binA, binB)
                st.session_state.result = cv2.compare(binA, binB, cv2.CMP_LT)
                st.session_state.result_name = "A < B"
            else:
                st.error("Carga ambas imágenes")
    
    with col3:
        if st.button("A == B", use_container_width=True):
            if imgA is not None and imgB is not None:
                binA, binB = binarize(imgA, umbral), binarize(imgB, umbral)
                binA, binB = ensure_same_size(binA, binB)
                st.session_state.result = cv2.compare(binA, binB, cv2.CMP_EQ)
                st.session_state.result_name = "A == B"
            else:
                st.error("Carga ambas imágenes")

# ========= TAB 4: COMPONENTES CONEXAS =========
with tab4:
    st.subheader("Análisis de Componentes Conexas")
    
    with st.expander("¿Qué son las componentes conexas?", expanded=False):
        st.markdown("""
        Las componentes conexas **identifican y etiquetan regiones conectadas** en una imagen binaria:
        
        - **Conectividad 4**: Considera solo vecinos horizontal y vertical (arriba, abajo, izq, der)
        - **Conectividad 8**: Considera todos los vecinos incluyendo diagonales
        
        Cada región conectada (objeto) recibe un **color único**, permitiendo:
        - Contar objetos automáticamente
        - Analizar forma y tamaño de cada objeto
        - Separar regiones individuales
        
        **Uso típico**: Conteo de células, detección de objetos, segmentación, análisis de formas.
        """)
    
    st.info("Identifica y colorea objetos conectados en una imagen binaria")
    
    if st.button("Etiquetar Componentes", use_container_width=False):
        # Seleccionar imagen fuente
        if cc_source == "A":
            src = imgA
        elif cc_source == "B":
            src = imgB
        else:  # Resultado
            src = st.session_state.result
        
        if src is None:
            st.error(f"No hay imagen en {cc_source}")
        else:
            # Binarizar si es necesario
            if len(src.shape) == 2:
                bin_img = (src > 0).astype(np.uint8) * 255
            else:
                bin_img = binarize(src, umbral)
            
            bin01 = (bin_img > 0).astype(np.uint8)
            
            # Aplicar componentes conexas
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                bin01, connectivity=cc_conn
            )
            
            if num_labels <= 1:
                st.warning("No se encontraron componentes conexas")
                st.session_state.result = np.zeros((*labels.shape, 3), dtype=np.uint8)
                st.session_state.result_name = f"Componentes (conn={cc_conn}) — 0 objetos"
            else:
                # Asignar colores aleatorios a cada componente
                rng = np.random.default_rng(42)
                colors = rng.integers(low=64, high=255, size=(num_labels, 3), dtype=np.uint8)
                colors[0] = (0, 0, 0)  # fondo negro
                color_labeled = colors[labels]
                
                objetos = num_labels - 1
                st.session_state.result = color_labeled
                st.session_state.result_name = f"Componentes (conn={cc_conn}) — {objetos} objetos"
                st.success(f"Se encontraron **{objetos}** componentes conexas")

# ========= Visualización de imágenes =========
st.markdown("---")
st.subheader("Visualización")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Imagen A**")
    if imgA is not None:
        st.image(imgA, use_container_width=True, clamp=True)
    else:
        st.info("Carga una imagen A en el sidebar")

with col2:
    st.markdown("**Imagen B**")
    if imgB is not None:
        st.image(imgB, use_container_width=True, clamp=True)
    else:
        st.info("Carga una imagen B en el sidebar")

with col3:
    result_name = st.session_state.get('result_name', 'Resultado')
    st.markdown(f"**{result_name}**")
    if st.session_state.result is not None:
        result_img = st.session_state.result
        # Convertir a RGB si es escala de grises
        if len(result_img.shape) == 2:
            result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2RGB)
        st.image(result_img, use_container_width=True, clamp=True)
    else:
        st.info("El resultado aparecerá aquí")

# Botones de control debajo de las imágenes
st.markdown("")
# Usar 5 columnas para posicionar el botón exactamente entre A y B
col_a, col_swap_ab, col_b, col_space, col_res = st.columns([2.5, 1, 2.5, 0.5, 3])

with col_a:
    # Espacio debajo de A
    st.markdown("")

with col_swap_ab:
    # Botón de intercambio A ↔ B (exactamente en medio de A y B)
    if st.button("⇄", use_container_width=True, key="swap_ab_center", help="Intercambiar A ↔ B"):
        if imgA is not None and imgB is not None:
            st.session_state.imgA, st.session_state.imgB = st.session_state.imgB, st.session_state.imgA
            st.rerun()
        else:
            st.warning("Carga A y B primero")

with col_b:
    # Espacio debajo de B
    st.markdown("")

with col_space:
    # Espacio entre B y Resultado
    st.markdown("")

with col_res:
    # Botones de asignación del resultado (debajo del Resultado)
    if st.session_state.result is not None:
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("← a A", use_container_width=True, key="result_to_a"):
                st.session_state.imgA = st.session_state.result.copy()
                st.success("Resultado copiado a A")
                st.rerun()
        with col_r2:
            if st.button("← a B", use_container_width=True, key="result_to_b"):
                st.session_state.imgB = st.session_state.result.copy()
                st.success("Resultado copiado a B")
                st.rerun()

# Opción de descarga
if st.session_state.result is not None:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        # Convertir a PIL para descargar
        result_to_save = st.session_state.result
        if len(result_to_save.shape) == 2:
            result_pil = Image.fromarray(result_to_save)
        else:
            result_pil = Image.fromarray(result_to_save)
        
        import io
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        
        st.download_button(
            label="Descargar Resultado",
            data=buf.getvalue(),
            file_name="resultado.png",
            mime="image/png",
            use_container_width=True
        )

st.markdown("---")
st.caption("Operaciones sobre Imágenes | Image Analysis 2025")

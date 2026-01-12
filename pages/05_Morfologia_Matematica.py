# -*- coding: utf-8 -*-
"""
Práctica 6: Morfología Matemática
Operaciones morfológicas en imágenes binarias y en escala de grises
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ========= Configuración de página =========
st.set_page_config(
    page_title="Morfología Matemática",
    page_icon="🔷",
    layout="wide"
)

st.markdown("""
<style>
    /* Industrial Tech Aesthetic - Charcoal & Amber */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap');

    :root {
        --bg-color: #0d0d0d;
        --card-bg: #161616;
        --border-color: #262626;
        --accent: #f59e0b;
        --text-bright: #ffffff;
    }

    header, [data-testid="stHeader"] { visibility: hidden; height: 0px; }
    .stApp { background-color: var(--bg-color); color: var(--text-bright); font-family: 'Space Grotesk', sans-serif; }
    [data-testid="stSidebar"] { background-color: #080808; border-right: 1px solid var(--border-color); }
    [data-testid="stSidebarNav"] li a span { color: #ffffff !important; font-weight: 600 !important; font-size: 1rem !important; }
    .main-header { font-size: 3rem; font-weight: 700; text-align: center; color: var(--accent); text-transform: uppercase; margin-bottom: 2rem; }
    .stButton > button { background-color: var(--accent) !important; color: black !important; font-weight: 700; border-radius: 2px !important; }
    .stTabs [data-baseweb="tab"] { color: #a3a3a3; }
    .stTabs [aria-selected="true"] { color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">MORPHOLOGY_05</div>', unsafe_allow_html=True)

# ========= Funciones auxiliares =========
def cargar_imagen(uploaded_file, modo='binaria'):
    """Carga imagen en modo binario o escala de grises."""
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('L')
        img = np.array(img)
        
        if modo == 'binaria':
            # Binarizar con umbral Otsu
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return img
    
    # Imagen sintética de prueba
    if modo == 'binaria':
        # Patrón binario con ruido
        img = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (200, 200), 255, -1)
        cv2.circle(img, (128, 128), 40, 0, -1)
        # Agregar ruido sal y pimienta
        noise = np.random.rand(256, 256)
        img[noise < 0.02] = 0
        img[noise > 0.98] = 255
    else:
        # Patrón en escala de grises
        img  = np.zeros((256, 256), dtype=np.uint8)
        for i in range(5):
            gray_val = 50 + i * 40
            x1, y1 = 30 + i*10, 30 + i*10
            x2, y2 = 220 - i*10, 220 - i*10
            cv2.rectangle(img, (x1, y1), (x2, y2), gray_val, -1)
    
    return img

def crear_kernel(forma, tamaño):
    """Crea elemento estructurante (kernel)."""
    if forma == 'Cuadrado':
        return np.ones((tamaño, tamaño), np.uint8)
    elif forma == 'Cruz':
        kernel = np.zeros((tamaño, tamaño), np.uint8)
        mid = tamaño // 2
        kernel[mid, :] = 1
        kernel[:, mid] = 1
        return kernel
    elif forma == 'Elipse':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamaño, tamaño))
    else:  # Círculo
        kernel = np.zeros((tamaño, tamaño), np.uint8)
        center = tamaño // 2
        cv2.circle(kernel, (center, center), center, 1, -1)
        return kernel

# Operaciones básicas (método tradicional)
def erosion_tradicional(img, kernel, iterations=1):
    """Erosión usando operador básico."""
    resultado = img.copy()
    for _ in range(iterations):
        resultado = cv2.erode(resultado, kernel, iterations=1)
    return resultado

def dilatacion_tradicional(img, kernel, iterations=1):
    """Dilatación usando operador básico."""
    resultado = img.copy()
    for _ in range(iterations):
        resultado = cv2.dilate(resultado, kernel, iterations=1)
    return resultado

def apertura_tradicional(img, kernel, iterations=1):
    """Apertura = Erosión + Dilatación."""
    erosionada = erosion_tradicional(img, kernel, iterations)
    apertura = dilatacion_tradicional(erosionada, kernel, iterations)
    return apertura

def cierre_tradicional(img, kernel, iterations=1):
    """Cierre = Dilatación + Erosión."""
    dilatada = dilatacion_tradicional(img, kernel, iterations)
    cierre = erosion_tradicional(dilatada, kernel, iterations)
    return cierre

# Operaciones avanzadas
def gradiente_morfologico(img, kernel):
    """Gradiente = Dilatación - Erosión."""
    dilatada = cv2.dilate(img, kernel, iterations=1)
    erosionada = cv2.erode(img, kernel, iterations=1)
    return cv2.subtract(dilatada, erosionada)

def top_hat(img, kernel):
    """Top Hat = Original - Apertura (resalta regiones brillantes)."""
    apertura = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(img, apertura)

def black_hat(img, kernel):
    """Black Hat = Cierre - Original (resalta regiones oscuras)."""
    cierre = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return cv2.subtract(cierre, img)

def frontera(img, kernel):
    """Frontera = Imagen - Erosión."""
    erosionada = cv2.erode(img, kernel, iterations=1)
    return cv2.subtract(img, erosionada)

# ========= Interfaz de Streamlit =========
st.title("Morfología Matemática")
st.markdown("### Operaciones morfológicas en imágenes binarias y en escala de grises")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Configuración")
    
    # Tipo de imagen
    tipo_imagen = st.radio("Tipo de imagen", ["Binaria", "Escala de Grises"])
    
    # Upload
    uploaded_file = st.file_uploader(
        "Cargar imagen (opcional)",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Si no se carga, se usará un patrón sintético"
    )
    
    if uploaded_file is None:
        st.info("Sin imagen = patrón sintético")
    
    st.markdown("---")
    
    # Parámetros del kernel
    st.subheader("Elemento Estructurante")
    forma_kernel = st.selectbox("Forma del kernel", ["Cuadrado", "Cruz", "Elipse", "Círculo"])
    tamaño_kernel = st.slider("Tamaño del kernel", 3, 15, 5, step=2)
    iteraciones = st.slider("Iteraciones", 1, 5, 1)
    
    st.markdown("---")
    
    # Operación a realizar
    st.subheader("Operación")
    categoria = st.radio("Categoría", ["Básicas", "Avanzadas"])

# Cargar imagen
modo = 'binaria' if tipo_imagen == "Binaria" else 'gris'
img_original = cargar_imagen(uploaded_file, modo)

# Crear kernel
kernel = crear_kernel(forma_kernel, tamaño_kernel)

# Visualizar kernel
with st.sidebar:
    with st.expander("Ver Elemento Estructurante"):
        fig_k, ax_k = plt.subplots(figsize=(3, 3))
        ax_k.imshow(kernel, cmap='gray', interpolation='nearest')
        ax_k.set_title(f'{forma_kernel} {tamaño_kernel}×{tamaño_kernel}')
        ax_k.axis('off')
        st.pyplot(fig_k)
        plt.close()

# ========= OPERACIONES BÁSICAS =========
if categoria == "Básicas":
    st.subheader("Operaciones Básicas")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Erosión", "Dilatación", "Apertura", "Cierre"])
    
    # TAB 1: EROSIÓN
    with tab1:
        st.markdown("**Erosión**: Reduce el área de las regiones blancas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Método Tradicional (`cv2.erode`)")
            img_erosion_trad = erosion_tradicional(img_original, kernel, iteraciones)
            st.image(img_erosion_trad, caption="Erosión Tradicional", use_container_width=True, clamp=True)
        
        with col2:
            st.markdown("##### Método OpenCV")
            img_erosion_cv = cv2.erode(img_original, kernel, iterations=iteraciones)
            st.image(img_erosion_cv, caption="Erosión OpenCV", use_container_width=True, clamp=True)
        
        # Verificar igualdad
        if np.array_equal(img_erosion_trad, img_erosion_cv):
            st.success("Ambos métodos producen el mismo resultado")
        
        with st.expander("Explicación de la Erosión"):
            st.markdown("""
            **Erosión**:
            - Reduce el tamaño de las regiones blancas (foreground)
            - Elimina píxeles en los bordes de los objetos
            - **Útil para**: Eliminar ruido pequeño, separar objetos unidos
            - **Efecto**: Adelgaza bordes, puede fragmentar objetos
            
            **Código tradicional**:
            ```python
            erosionada = cv2.erode(imagen, kernel, iterations=1)
            ```
            """)
    
    # TAB 2: DILATACIÓN
    with tab2:
        st.markdown("**Dilatación**: Incrementa el área de las regiones blancas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Método Tradicional (`cv2.dilate`)")
            img_dilate_trad = dilatacion_tradicional(img_original, kernel, iteraciones)
            st.image(img_dilate_trad, caption="Dilatación Tradicional", use_container_width=True, clamp=True)
        
        with col2:
            st.markdown("##### Método OpenCV")
            img_dilate_cv = cv2.dilate(img_original, kernel, iterations=iteraciones)
            st.image(img_dilate_cv, caption="Dilatación OpenCV", use_container_width=True, clamp=True)
        
        if np.array_equal(img_dilate_trad, img_dilate_cv):
            st.success("Ambos métodos producen el mismo resultado")
        
        with st.expander("Explicación de la Dilatación"):
            st.markdown("""
            **Dilatación**:
            - Incrementa el tamaño de las regiones blancas
            - Agrega píxeles en los bordes de los objetos
            - **Útil para**: Rellenar agujeros pequeños, unir objetos cercanos
            - **Efecto**: Engrosa bordes, une objetos separados
            
            **Código tradicional**:
            ```python
            dilatada = cv2.dilate(imagen, kernel, iterations=1)
            ```
            """)
    
    # TAB 3: APERTURA
    with tab3:
        st.markdown("**Apertura**: Erosión seguida de Dilatación (elimina ruido)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Método Tradicional (Erosión + Dilatación)")
            img_open_trad = apertura_tradicional(img_original, kernel, iteraciones)
            st.image(img_open_trad, caption="Apertura Tradicional", use_container_width=True, clamp=True)
            
            with st.expander("Ver código tradicional"):
                st.code("""
erosionada = cv2.erode(imagen, kernel, iterations=1)
apertura = cv2.dilate(erosionada, kernel, iterations=1)
                """, language="python")
        
        with col2:
            st.markdown("##### Método OpenCV (`MORPH_OPEN`)")
            img_open_cv = cv2.morphologyEx(img_original, cv2.MORPH_OPEN, kernel, iterations=iteraciones)
            st.image(img_open_cv, caption="Apertura OpenCV", use_container_width=True, clamp=True)
            
            with st.expander("Ver código OpenCV"):
                st.code("""
apertura = cv2.morphologyEx(imagen, 
                            cv2.MORPH_OPEN, 
                            kernel, 
                            iterations=1)
                """, language="python")
        
        if np.array_equal(img_open_trad, img_open_cv):
            st.success("Ambos métodos producen el mismo resultado")
        
        with st.expander("Explicación de la Apertura"):
            st.markdown("""
            **Apertura (Opening)**:
            - **Secuencia**: Erosión → Dilatación
            - Elimina ruido pequeño sin cambiar mucho el tamaño de objetos grandes
            - **Útil para**: Eliminar puntos de ruido, suavizar bordes
            - **Preserva**: Forma general de objetos grandes
            """)
    
    # TAB 4: CIERRE
    with tab4:
        st.markdown("**Cierre**: Dilatación seguida de Erosión (rellena agujeros)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Método Tradicional (Dilatación + Erosión)")
            img_close_trad = cierre_tradicional(img_original, kernel, iteraciones)
            st.image(img_close_trad, caption="Cierre Tradicional", use_container_width=True, clamp=True)
            
            with st.expander("Ver código tradicional"):
                st.code("""
dilatada = cv2.dilate(imagen, kernel, iterations=1)
cierre = cv2.erode(dilatada, kernel, iterations=1)
                """, language="python")
        
        with col2:
            st.markdown("##### Método OpenCV (`MORPH_CLOSE`)")
            img_close_cv = cv2.morphologyEx(img_original, cv2.MORPH_CLOSE, kernel, iterations=iteraciones)
            st.image(img_close_cv, caption="Cierre OpenCV", use_container_width=True, clamp=True)
            
            with st.expander("Ver código OpenCV"):
                st.code("""
cierre = cv2.morphologyEx(imagen, 
                          cv2.MORPH_CLOSE, 
                          kernel, 
                          iterations=1)
                """, language="python")
        
        if np.array_equal(img_close_trad, img_close_cv):
            st.success("Ambos métodos producen el mismo resultado")
        
        with st.expander("Explicación del Cierre"):
            st.markdown("""
            **Cierre (Closing)**:
            - **Secuencia**: Dilatación → Erosión
            - Rellena agujeros pequeños y conecta regiones cercanas
            - **Útil para**: Cerrar gaps, rellenar huecos, unir objetos
            - **Preserva**: Tamaño aproximado de objetos
            """)

# ========= OPERACIONES AVANZADAS =========
else:  # Avanzadas
    st.subheader("Operaciones Avanzadas")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Gradiente", "Top Hat", "Black Hat", "Frontera"])
    
    # TAB 1: GRADIENTE MORFOLÓGICO
    with tab1:
        st.markdown("**Gradiente Morfológico**: Dilatación - Erosión (resalta bordes)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### Original")
            st.image(img_original, caption="Imagen Original", use_container_width=True, clamp=True)
        
        with col2:
            st.markdown("##### Gradiente (Tradicional)")
            img_grad_trad = gradiente_morfologico(img_original, kernel)
            st.image(img_grad_trad, caption="Gradiente", use_container_width=True, clamp=True)
        
        with col3:
            st.markdown("##### Gradiente (OpenCV)")
            img_grad_cv = cv2.morphologyEx(img_original, cv2.MORPH_GRADIENT, kernel)
            st.image(img_grad_cv, caption="Gradiente OpenCV", use_container_width=True, clamp=True)
        
        with st.expander("Explicación del Gradiente"):
            st.markdown("""
            **Gradiente Morfológico**:
            - **Fórmula**: Dilatación(I) - Erosión(I)
            - Resalta los **bordes** de los objetos
            - **Útil para**: Detección de contornos, segmentación
            
            ```python
            dilatada = cv2.dilate(imagen, kernel)
            erosionada = cv2.erode(imagen, kernel)
            gradiente = cv2.subtract(dilatada, erosionada)
            ```
            """)
    
    # TAB 2: TOP HAT
    with tab2:
        st.markdown("**Top Hat**: Original - Apertura (resalta puntos brillantes)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### Original")
            st.image(img_original, caption="Imagen Original", use_container_width=True, clamp=True)
        
        with col2:
            st.markdown("##### Top Hat (Tradicional)")
            img_tophat_trad = top_hat(img_original, kernel)
            st.image(img_tophat_trad, caption="Top Hat", use_container_width=True, clamp=True)
        
        with col3:
            st.markdown("##### Top Hat (OpenCV)")
            img_tophat_cv = cv2.morphologyEx(img_original, cv2.MORPH_TOPHAT, kernel)
            st.image(img_tophat_cv, caption="Top Hat OpenCV", use_container_width=True, clamp=True)
        
        with st.expander("Explicación de Top Hat"):
            st.markdown("""
            **Top Hat (Sombrero de Copa)**:
            - **Fórmula**: Original - Apertura
            - Resalta **regiones brillantes** más pequeñas que el elemento estructurante
            - **Útil para**: Detectar puntos brillantes, eliminar iluminación no uniforme
            
            ```python
            apertura = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)
            tophat = cv2.subtract(imagen, apertura)
            ```
            """)
    
    # TAB 3: BLACK HAT
    with tab3:
        st.markdown("**Black Hat**: Cierre - Original (resalta puntos oscuros)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### Original")
            st.image(img_original, caption="Imagen Original", use_container_width=True, clamp=True)
        
        with col2:
            st.markdown("##### Black Hat (Tradicional)")
            img_blackhat_trad = black_hat(img_original, kernel)
            st.image(img_blackhat_trad, caption="Black Hat", use_container_width=True, clamp=True)
        
        with col3:
            st.markdown("##### Black Hat (OpenCV)")
            img_blackhat_cv = cv2.morphologyEx(img_original, cv2.MORPH_BLACKHAT, kernel)
            st.image(img_blackhat_cv, caption="Black Hat OpenCV", use_container_width=True, clamp=True)
        
        with st.expander("Explicación de Black Hat"):
            st.markdown("""
            **Black Hat (Bot Hat)**:
            - **Fórmula**: Cierre - Original
            - Resalta **regiones oscuras** más pequeñas que el elemento estructurante
            - **Útil para**: Detectar valles oscuros, agujeros
            
            ```python
            cierre = cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)
            blackhat = cv2.subtract(cierre, imagen)
            ```
            """)
    
    # TAB 4: FRONTERA
    with tab4:
        st.markdown("**Frontera**: Imagen - Erosión (extrae el contorno)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Original")
            st.image(img_original, caption="Imagen Original", use_container_width=True, clamp=True)
        
        with col2:
            st.markdown("##### Frontera")
            img_frontera = frontera(img_original, kernel)
            st.image(img_frontera, caption="Frontera", use_container_width=True, clamp=True)
        
        with st.expander("Explicación de la Frontera"):
            st.markdown("""
            **Frontera (Boundary)**:
            - **Fórmula**: Original - Erosión
            - Extrae el **contorno** de los objetos
            - **Útil para**: Análisis de formas, detección de bordes
            
            ```python
            erosionada = cv2.erode(imagen, kernel)
            frontera = cv2.subtract(imagen, erosionada)
            ```
            """)

# ========= Comparación lado a lado =========
st.markdown("---")
st.subheader("Original vs Resultado")

col_orig, col_res = st.columns(2)

with col_orig:
    st.markdown("**Imagen Original**")
    st.image(img_original, caption=f"Original ({tipo_imagen})", use_container_width=True, clamp=True)

with col_res:
    st.markdown("**Vista Previa**")
    st.info("Explora las operaciones en las tabs de arriba")

# Footer con tabla comparativa
st.markdown("---")
with st.expander("Tabla Comparativa: Binaria vs Escala de Grises"):
    st.markdown("""
    | Aspecto | Morfología Binaria | Morfología en Escala de Grises |
    |---------|-------------------|--------------------------------|
    | **Definición** | Opera sobre píxeles 0/255 | Opera sobre píxeles 0-255 |
    | **Operaciones** | Lógica binaria (unión/intersección) | Operaciones min/max |
    | **Erosión** | Elimina píxeles en bordes | Reduce intensidades locales |
    | **Dilatación** | Agrega píxeles en bordes | Aumenta intensidades locales |
    | **Resultado** | Imagen sigue siendo binaria | Imagen sigue en escala de grises |
    | **Uso típico** | Segmentación, análisis de formas | Suavizado, realce de contraste |
    """)

with st.expander("Objetivos de Aprendizaje"):
    st.markdown("""
    Al completar esta práctica, serás capaz de:
    
    - Aplicar **erosión** y **dilatación** en imágenes binarias y grises
    - Implementar **apertura** y **cierre** (método tradicional y OpenCV)
    - Calcular el **gradiente morfológico** para detección de bordes
    - Usar **Top Hat** y **Black Hat** para realce selectivo
    - Extraer **fronteras** de objetos
    - Diseñar elementos estructurantes apropiados
    - Elegir la operación correcta según el problema
    """)

st.caption("Morfología Matemática | Image Analysis 2025")

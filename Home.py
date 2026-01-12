import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Spectra - Image Analysis",
    page_icon="image.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo morado con degradado
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
        --text-subtle: #a3a3a3;
    }

    /* Hide Streamlit Header */
    header, [data-testid="stHeader"] { visibility: hidden; height: 0px; }

    .stApp {
        background-color: var(--bg-color);
        color: var(--text-bright);
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Sidebar - High Contrast */
    [data-testid="stSidebar"] {
        background-color: #080808;
        border-right: 1px solid var(--border-color);
    }
    
    /* Sidebar Navigation Links - WHITE & BOLD */
    [data-testid="stSidebarNav"] li a span {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Sidebar Section Titles - AMBER */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--accent) !important;
    }

    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: -1px;
    }

    .glass-card {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        padding: 24px;
        margin-bottom: 24px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar con branding Spectra
with st.sidebar:
    st.image("image.png", width=180)
    st.markdown("<h2 style='text-align: center;'>SPECTRA</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a3a3a3; font-size: 12px; letter-spacing: 2px;'>IMAGE ANALYSIS ENGINE</p>", unsafe_allow_html=True)
    st.markdown("---")

# Título de bienvenida
st.markdown('<div class="main-header">SPECTRA PROJECT</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: var(--text-subtle); letter-spacing: 2px; margin-bottom: 3rem;">COMPUTER VISION ACADEMIC PLATFORM</p>', unsafe_allow_html=True)

# Bienvenida
st.markdown("""
## ¡Bienvenido!

Este proyecto recopila las **prácticas de Image Analysis** desarrolladas durante el semestre.
Cada módulo está implementado como una página interactiva de Streamlit.

### Nuevo Diseño
Ahora con **tema morado con degradado** para una mejor experiencia visual.
""")

st.markdown("---")

# Módulos disponibles
st.subheader("Módulos Disponibles")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 01 - Procesamiento de Imágenes
    - Modelos de color (RGB, YIQ, CMY, HSV)
    - Canales RGB con realce
    - Binarización ajustable
    - Histogramas interactivos
    
    #### 02 - Operaciones sobre Imágenes
    - Operaciones aritméticas
    - Operaciones lógicas
    - Operaciones relacionales
    - Componentes conexas
    
    #### 03 - Pseudocolor
    - 7 Colormaps de OpenCV
    - Colormap pastel personalizado
    - Ajustes HSV avanzados
    - Exportación en ZIP
    """)

with col2:
    st.markdown("""
    #### 04 - Procesamiento en Frecuencia
    - **FFT**: Filtros Ideal, Gaussiano, Butterworth
    - **DCT**: Compresión tipo JPEG
    - Métricas PSNR
    - Comparación de calidades
    
    #### 05 - Morfología Matemática
    - Erosión y Dilatación
    - Apertura y Cierre
    - Gradiente morfológico
    - Top Hat y Black Hat
    - Componentes conexas
    
    #### 06 - Clasificación de Frutas
    - Detección de tipo: 🍎, 🍌, 🍊
    - Calidad: Fresca vs Podrida
    - Segmentación clásica (GrabCut/HSV)
    - Basado en Deep Learning (MobileNetV2)
    """)

st.markdown("---")

# Instrucciones
st.subheader("Cómo usar")
st.markdown("""
1. **Selecciona un módulo** en el sidebar izquierdo
2. **Carga una imagen** usando el botón de upload
3. **Ajusta los parámetros** con los controles interactivos
4. **Visualiza los resultados** en tiempo real
5. **Descarga** las imágenes procesadas

**Tip**: Cada módulo incluye explicaciones y ejemplos educativos.
""")

st.markdown("---")

# Información adicional
col1, col2 = st.columns(2)

with col1:
    st.info("**Tip**: Cada módulo es independiente y puede ejecutarse por separado.")

with col2:
    st.success("Explora los diferentes programas en el menú lateral")

# Footer
st.markdown("---")

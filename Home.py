import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Proyecto Image Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo morado con degradado
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a0b2e 0%, #2d1b4e 50%, #16213e 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0324 0%, #1a0b2e 100%);
    }
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, label {
        color: #ede9fe !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #6b46c1 0%, #9f7aea 100%);
        color: white;
        border: none;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #553c9a 0%, #805ad5 100%);
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.title("🔬 Proyecto de Image Analysis")
st.markdown("---")

# Bienvenida
st.markdown("""
## 👋 ¡Bienvenido!

Este proyecto recopila las **prácticas de Image Analysis** desarrolladas durante el semestre.
Cada módulo está implementado como una página interactiva de Streamlit.

### 🎨 Nuevo Diseño
Ahora con **tema morado con degradado** para una mejor experiencia visual.
""")

st.markdown("---")

# Módulos disponibles
st.subheader("📚 Módulos Disponibles")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 🎨 01 - Procesamiento de Imágenes
    - Modelos de color (RGB, YIQ, CMY, HSV)
    - Canales RGB con realce
    - Binarización ajustable
    - Histogramas interactivos
    
    #### 🔧 02 - Operaciones sobre Imágenes
    - Operaciones aritméticas
    - Operaciones lógicas
    - Operaciones relacionales
    - Componentes conexas
    
    #### 🌈 03 - Pseudocolor
    - 7 Colormaps de OpenCV
    - Colormap pastel personalizado
    - Ajustes HSV avanzados
    - Exportación en ZIP
    """)

with col2:
    st.markdown("""
    #### 📊 04 - Procesamiento en Frecuencia
    - **FFT**: Filtros Ideal, Gaussiano, Butterworth
    - **DCT**: Compresión tipo JPEG
    - Métricas PSNR
    - Comparación de calidades
    
    #### 🔷 05 - Morfología Matemática
    - Erosión y Dilatación
    - Apertura y Cierre
    - Gradiente morfológico
    - Top Hat y Black Hat
    - Componentes conexas
    """)

st.markdown("---")

# Instrucciones
st.subheader("🚀 Cómo usar")
st.markdown("""
1. **Selecciona un módulo** en el sidebar izquierdo
2. **Carga una imagen** usando el botón de upload
3. **Ajusta los parámetros** con los controles interactivos
4. **Visualiza los resultados** en tiempo real
5. **Descarga** las imágenes procesadas

💡 **Tip**: Cada módulo incluye explicaciones y ejemplos educativos.
""")

st.markdown("---")

# Información adicional
col1, col2 = st.columns(2)

with col1:
    st.info("💡 **Tip**: Cada módulo es independiente y puede ejecutarse por separado.")

with col2:
    st.success("✅ Explora los diferentes programas en el menú lateral")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #c4b5fd;'>
    <p>Proyecto Image Analysis - Semestre 2025</p>
</div>
""", unsafe_allow_html=True)

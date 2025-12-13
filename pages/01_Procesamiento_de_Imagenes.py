# -*- coding: utf-8 -*-
"""
Procesamiento de Imágenes y Conversión de Modelos de Color
Adaptado a Streamlit
"""
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

# ========= Configuración de página =========
st.set_page_config(
    page_title="Procesamiento de Imágenes",
    page_icon="",
    layout="wide"
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
    .stMarkdown, .stText, p, h1, h2, h3, label {
        color: #ede9fe !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #6b46c1 0%, #9f7aea 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ========= Utilidades generales =========
def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convierte cualquier imagen a uint8 sin alterar su apariencia visual."""
    if img.dtype == np.uint8:
        return img
    if img.dtype.kind == "f" and 0.0 <= float(img.min()) and float(img.max()) <= 1.0:
        return (img * 255).clip(0, 255).astype(np.uint8)
    m, M = float(img.min()), float(img.max())
    if M - m < 1e-12:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - m) / (M - m) * 255.0).astype(np.uint8)


def imagen_a_bytes(img: np.ndarray, nombre: str = "imagen.png") -> bytes:
    """Convierte una imagen numpy array a bytes PNG para descarga."""
    # Asegurar que sea uint8
    if img.dtype != np.uint8:
        img = to_uint8(img)
    
    # Convertir a PIL Image
    if img.ndim == 2:
        # Escala de grises
        pil_img = Image.fromarray(img, mode='L')
    else:
        # RGB
        pil_img = Image.fromarray(img, mode='RGB')
    
    # Convertir a bytes
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return buf.getvalue()

def plot_histogram(img_u8: np.ndarray, title: str):
    """Crea un histograma interactivo con Plotly: 1 curva si es gris, 3 curvas si es RGB."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    if img_u8.ndim == 2:  # escala de grises
        hist, bins = np.histogram(img_u8.ravel(), bins=256, range=(0, 255))
        fig.add_trace(go.Scatter(
            x=bins[:-1],
            y=hist,
            mode='lines',
            name='Intensidad',
            line=dict(color='#9CA3AF', width=2),
            fill='tozeroy',
            fillcolor='rgba(156, 163, 175, 0.3)',
            hovertemplate='Nivel: %{x}<br>Frecuencia: %{y}<extra></extra>'
        ))
    else:  # RGB
        colors = {
            'red': ('#EF4444', 'rgba(239, 68, 68, 0.2)'),
            'green': ('#10B981', 'rgba(16, 185, 129, 0.2)'),
            'blue': ('#3B82F6', 'rgba(59, 130, 246, 0.2)')
        }
        
        for i, (channel, (line_color, fill_color)) in enumerate(colors.items()):
            hist, bins = np.histogram(img_u8[..., i].ravel(), bins=256, range=(0, 255))
            fig.add_trace(go.Scatter(
                x=bins[:-1],
                y=hist,
                mode='lines',
                name=channel.upper(),
                line=dict(color=line_color, width=2),
                fill='tozeroy',
                fillcolor=fill_color,
                hovertemplate=f'{channel.upper()}<br>Nivel: %{{x}}<br>Frecuencia: %{{y}}<extra></extra>'
            ))
    
    # Configuración del layout con tema oscuro
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color='#E5E7EB')
        ),
        xaxis=dict(
            title='Intensidad',
            range=[0, 255],
            gridcolor='rgba(255, 255, 255, 0.1)',
            color='#E5E7EB'
        ),
        yaxis=dict(
            title='Frecuencia',
            gridcolor='rgba(255, 255, 255, 0.1)',
            color='#E5E7EB'
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#E5E7EB')
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=300
    )
    
    return fig


# ========= Funciones de conversión =========
def cargar_rgb(uploaded_file) -> np.ndarray:
    """Carga imagen desde archivo subido."""
    pil_image = Image.open(uploaded_file).convert("RGB")
    return np.array(pil_image)


def separar_canales(rgb: np.ndarray):
    """Devuelve canales R, G, B con realce de color respectivo."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    
    # Crear imágenes con realce de color
    r_color = np.zeros_like(rgb)
    r_color[..., 0] = r  # Solo canal rojo activo
    
    g_color = np.zeros_like(rgb)
    g_color[..., 1] = g  # Solo canal verde activo
    
    b_color = np.zeros_like(rgb)
    b_color[..., 2] = b  # Solo canal azul activo
    
    return r_color, g_color, b_color


def a_grises_bt601(rgb: np.ndarray) -> np.ndarray:
    """Escala de grises por luminancia (BT.601) con OpenCV."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def binarizar(gray_u8: np.ndarray, t: int = 128) -> np.ndarray:
    """Umbral fijo."""
    _, binaria = cv2.threshold(gray_u8, t, 255, cv2.THRESH_BINARY)
    return binaria


def to_float01(rgb_u8: np.ndarray) -> np.ndarray:
    """Convierte RGB uint8 a float32 [0,1]."""
    return rgb_u8.astype(np.float32) / 255.0


def convertir_yiq(rgb_u8: np.ndarray) -> dict:
    """YIQ: devuelve Y, I, Q como 2D usando matriz de transformación."""
    rgb_f = to_float01(rgb_u8)
    
    # Matriz de transformación RGB a YIQ (estándar NTSC)
    transform_matrix = np.array([
        [0.299, 0.587, 0.114],      # Y
        [0.5959, -0.2746, -0.3213],  # I
        [0.2115, -0.5227, 0.3112]    # Q
    ])
    
    # Aplicar transformación
    yiq = np.dot(rgb_f, transform_matrix.T)
    Y, I, Q = yiq[..., 0], yiq[..., 1], yiq[..., 2]

    def rescale01(x):
        xmin, xmax = float(x.min()), float(x.max())
        if xmax - xmin < 1e-12:
            return np.zeros_like(x, dtype=np.float32)
        return (x - xmin) / (xmax - xmin)

    Y_u8 = to_uint8(Y)
    I_u8 = to_uint8(rescale01(I))
    Q_u8 = to_uint8(rescale01(Q))

    return {
        "Y (luminancia)": Y_u8,
        "I (crominancia)": I_u8,
        "Q (crominancia)": Q_u8
    }


def convertir_cmy(rgb_u8: np.ndarray) -> dict:
    """CMY sustractivo: C=255-R, M=255-G, Y=255-B."""
    cmy_u8 = 255 - rgb_u8
    C, M, Y = cmy_u8[..., 0], cmy_u8[..., 1], cmy_u8[..., 2]
    return {
        "C (255-R)": C,
        "M (255-G)": M,
        "Y (255-B)": Y
    }


def convertir_hsv(rgb_u8: np.ndarray) -> dict:
    """HSV: H, S, V usando OpenCV."""
    # OpenCV usa BGR, así que convertimos de RGB a BGR primero
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    # Convertir a HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return {
        "H (matiz)": H,
        "S (saturación)": S,
        "V (valor)": V
    }


# ========= Interfaz de Streamlit =========
st.title("Procesamiento de Imágenes")
st.markdown("### Conversión de Modelos de Color y Análisis de Histogramas")
st.markdown("---")

# Sidebar para controles
with st.sidebar:
    st.header("Configuración")
    
    # Upload de imagen
    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Sube una imagen para procesar"
    )
    
    # Control de umbral para binarización
    umbral = st.slider(
        "Umbral de binarización",
        min_value=0,
        max_value=255,
        value=128,
        help="Ajusta el umbral para la binarización"
    )
    
    st.markdown("---")
    st.markdown("**Opciones de visualización:**")
    mostrar_histogramas = st.checkbox("Mostrar histogramas", value=True)

# Procesamiento principal
if uploaded_file is not None:
    # Cargar imagen
    rgb = cargar_rgb(uploaded_file)
    
    # Tabs para organizar el contenido
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Original",
        "RGB Canales",
        "Escala Grises",
        "Binarización",
        "YIQ",
        "CMY",
        "HSV"
    ])
    
    # Tab 1: Imagen original
    with tab1:
        st.subheader("Imagen Original RGB")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(rgb, caption="Original RGB", use_container_width=True)
            st.download_button(
                label="Descargar Original RGB",
                data=imagen_a_bytes(rgb),
                file_name="original_rgb.png",
                mime="image/png",
                use_container_width=True
            )
        with col2:
            if mostrar_histogramas:
                st.plotly_chart(plot_histogram(rgb, "Histograma RGB"), use_container_width=True)
    
    # Tab 2: Canales RGB
    with tab2:
        st.subheader("Canales RGB Separados")
        r_color, g_color, b_color = separar_canales(rgb)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(r_color, caption="Canal R (realce rojo)", use_container_width=True, clamp=True)
            st.download_button(
                label="Descargar Canal R",
                data=imagen_a_bytes(r_color),
                file_name="canal_rojo.png",
                mime="image/png",
                use_container_width=True
            )
            if mostrar_histogramas:
                # Extraer solo el canal rojo para el histograma
                st.plotly_chart(plot_histogram(r_color[..., 0], "Histograma Canal R"), use_container_width=True)
        with col2:
            st.image(g_color, caption="Canal G (realce verde)", use_container_width=True, clamp=True)
            st.download_button(
                label="Descargar Canal G",
                data=imagen_a_bytes(g_color),
                file_name="canal_verde.png",
                mime="image/png",
                use_container_width=True
            )
            if mostrar_histogramas:
                # Extraer solo el canal verde para el histograma
                st.plotly_chart(plot_histogram(g_color[..., 1], "Histograma Canal G"), use_container_width=True)
        with col3:
            st.image(b_color, caption="Canal B (realce azul)", use_container_width=True, clamp=True)
            st.download_button(
                label="Descargar Canal B",
                data=imagen_a_bytes(b_color),
                file_name="canal_azul.png",
                mime="image/png",
                use_container_width=True
            )
            if mostrar_histogramas:
                # Extraer solo el canal azul para el histograma
                st.plotly_chart(plot_histogram(b_color[..., 2], "Histograma Canal B"), use_container_width=True)
    
    # Tab 3: Escala de grises
    with tab3:
        st.subheader("Escala de Grises (BT.601)")
        gray = a_grises_bt601(rgb)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(gray, caption="Escala de grises", use_container_width=True, clamp=True)
            st.download_button(
                label="Descargar Escala de Grises",
                data=imagen_a_bytes(gray),
                file_name="escala_grises.png",
                mime="image/png",
                use_container_width=True
            )
        with col2:
            if mostrar_histogramas:
                st.plotly_chart(plot_histogram(gray, "Histograma Escala de Grises"), use_container_width=True)
    
    # Tab 4: Binarización
    with tab4:
        st.subheader(f"Binarización (Umbral = {umbral})")
        binaria = binarizar(gray, umbral)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(binaria, caption=f"Binarizada (t={umbral})", use_container_width=True, clamp=True)
            st.download_button(
                label="Descargar Binarización",
                data=imagen_a_bytes(binaria),
                file_name=f"binarizada_t{umbral}.png",
                mime="image/png",
                use_container_width=True
            )
        with col2:
            if mostrar_histogramas:
                st.plotly_chart(plot_histogram(binaria, "Histograma Binarización"), use_container_width=True)
            st.info(f"Ajusta el umbral en el sidebar (actual: {umbral})")
    
    # Tab 5: YIQ
    with tab5:
        st.subheader("Modelo YIQ")
        yiq = convertir_yiq(rgb)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(yiq["Y (luminancia)"], caption="Y (luminancia)", use_container_width=True, clamp=True)
            st.download_button(
                label="Descargar Y",
                data=imagen_a_bytes(yiq["Y (luminancia)"]),
                file_name="yiq_luminancia.png",
                mime="image/png",
                use_container_width=True
            )
            if mostrar_histogramas:
                st.plotly_chart(plot_histogram(yiq["Y (luminancia)"], "Histograma Y"), use_container_width=True)
        with col2:
            st.image(yiq["I (crominancia)"], caption="I (crominancia)", use_container_width=True, clamp=True)
            st.download_button(
                label="Descargar I",
                data=imagen_a_bytes(yiq["I (crominancia)"]),
                file_name="yiq_crominancia_i.png",
                mime="image/png",
                use_container_width=True
            )
            if mostrar_histogramas:
                st.plotly_chart(plot_histogram(yiq["I (crominancia)"], "Histograma I"), use_container_width=True)
        with col3:
            st.image(yiq["Q (crominancia)"], caption="Q (crominancia)", use_container_width=True, clamp=True)
            st.download_button(
                label="Descargar Q",
                data=imagen_a_bytes(yiq["Q (crominancia)"]),
                file_name="yiq_crominancia_q.png",
                mime="image/png",
                use_container_width=True
            )
            if mostrar_histogramas:
                st.plotly_chart(plot_histogram(yiq["Q (crominancia)"], "Histograma Q"), use_container_width=True)
    
    # Tab 6: CMY
    with tab6:
        st.subheader("Modelo CMY (Sustractivo)")
        cmy = convertir_cmy(rgb)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(cmy["C (255-R)"], caption="C (255-R)", use_container_width=True, clamp=True)
            st.download_button(
                label="Descargar C",
                data=imagen_a_bytes(cmy["C (255-R)"]),
                file_name="cmy_cyan.png",
                mime="image/png",
                use_container_width=True
            )
            if mostrar_histogramas:
                st.plotly_chart(plot_histogram(cmy["C (255-R)"], "Histograma C"), use_container_width=True)
        with col2:
            st.image(cmy["M (255-G)"], caption="M (255-G)", use_container_width=True, clamp=True)
            st.download_button(
                label="Descargar M",
                data=imagen_a_bytes(cmy["M (255-G)"]),
                file_name="cmy_magenta.png",
                mime="image/png",
                use_container_width=True
            )
            if mostrar_histogramas:
                st.plotly_chart(plot_histogram(cmy["M (255-G)"], "Histograma M"), use_container_width=True)
        with col3:
            st.image(cmy["Y (255-B)"], caption="Y (255-B)", use_container_width=True, clamp=True)
            st.download_button(
                label="Descargar Y",
                data=imagen_a_bytes(cmy["Y (255-B)"]),
                file_name="cmy_yellow.png",
                mime="image/png",
                use_container_width=True
            )
            if mostrar_histogramas:
                st.plotly_chart(plot_histogram(cmy["Y (255-B)"], "Histograma Y"), use_container_width=True)
    
    # Tab 7: HSV
    with tab7:
        st.subheader("Modelo HSV")
        hsv = convertir_hsv(rgb)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(hsv["H (matiz)"], caption="H (matiz)", use_container_width=True, clamp=True)
            st.download_button(
                label="Descargar H",
                data=imagen_a_bytes(hsv["H (matiz)"]),
                file_name="hsv_hue.png",
                mime="image/png",
                use_container_width=True
            )
            if mostrar_histogramas:
                st.plotly_chart(plot_histogram(hsv["H (matiz)"], "Histograma H"), use_container_width=True)
        with col2:
            st.image(hsv["S (saturación)"], caption="S (saturación)", use_container_width=True, clamp=True)
            st.download_button(
                label="Descargar S",
                data=imagen_a_bytes(hsv["S (saturación)"]),
                file_name="hsv_saturation.png",
                mime="image/png",
                use_container_width=True
            )
            if mostrar_histogramas:
                st.plotly_chart(plot_histogram(hsv["S (saturación)"], "Histograma S"), use_container_width=True)
        with col3:
            st.image(hsv["V (valor)"], caption="V (valor)", use_container_width=True, clamp=True)
            st.download_button(
                label="Descargar V",
                data=imagen_a_bytes(hsv["V (valor)"]),
                file_name="hsv_value.png",
                mime="image/png",
                use_container_width=True
            )
            if mostrar_histogramas:
                st.plotly_chart(plot_histogram(hsv["V (valor)"], "Histograma V"), use_container_width=True)

else:
    # Instrucciones cuando no hay imagen
    st.info("Por favor, sube una imagen usando el sidebar para comenzar el procesamiento")
    
    st.markdown("""
    ### Funcionalidades disponibles:
    
    - **Original RGB**: Visualización de la imagen cargada
    - **Canales RGB**: Separación en canales Rojo, Verde y Azul
    - **Escala de Grises**: Conversión usando el estándar BT.601
    - **Binarización**: Conversión a imagen binaria con umbral ajustable
    - **Modelo YIQ**: Separación en luminancia (Y) y crominancia (I, Q)
    - **Modelo CMY**: Modelo sustractivo de color (Cyan, Magenta, Yellow)
    - **Modelo HSV**: Matiz, Saturación y Valor
    
    Cada conversión incluye visualización de histogramas opcionales.
    """)

st.markdown("---")
st.caption("Procesamiento de Imágenes | Image Analysis 2025")

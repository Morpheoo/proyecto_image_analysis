# practica4.py  (UI Streamlit con controles de opacidad/saturación)
# -------------------------------------------------------
# - Carga imagen (o gradiente)
# - Colormaps OpenCV + Pastel personalizado
# - Sliders para Gamma (pre), Brightness del Pastel, Saturación, Valor, Mezcla con grises
# - Descarga individual y ZIP
# - Comparativa robusta (padding por fila)
# -------------------------------------------------------

import io
import zipfile
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

# -------------- Apariencia básica --------------
st.set_page_config(page_title="Pseudocolor UI | Práctica 4", layout="wide")
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

st.markdown('<div class="main-header">PSEUDOCOLOR_03</div>', unsafe_allow_html=True)

# -------------- Utilidades --------------
def to_gray_array(img_file, width=768):
    """Convierte a escala de grises y redimensiona manteniendo aspecto.
    Si no hay imagen, devuelve un gradiente de prueba."""
    if img_file is not None:
        img = Image.open(img_file).convert("L")
        w, h = img.size
        new_w = width
        new_h = max(1, int(h * (new_w / w)))
        img = img.resize((new_w, new_h), Image.BICUBIC)
        return np.array(img, dtype=np.uint8)
    # Gradiente fallback (256 x width)
    grad = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (256, 1))
    return grad

def apply_opencv_colormap(gray, cmap_flag):
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cmap_flag)  # BGR

def pastel_colormap(brightness=1.0):
    """Pastel en 0–255 -> normaliza a 0–1 y ajusta brillo.
    brightness <1 oscurece; >1 aclara (se clippea a 1)."""
    colores_255 = np.array([
        [255, 204, 230],  # rosa claro
        [204, 255, 204],  # verde menta
        [204, 230, 255],  # azul lavanda
        [255, 255, 204],  # amarillo suave
        [230, 204, 255],  # violeta claro
    ], dtype=np.float32)
    cols = (colores_255 / 255.0) * float(brightness)
    cols = np.clip(cols, 0.0, 1.0)
    return LinearSegmentedColormap.from_list("PastelMap", cols.tolist(), N=256)

def gothic_colormap():
    """Mapa de colores gótico: Negro violeta profundo -> Rojo sangre -> Rosado oscuro
    Con suavizado para transiciones graduales."""
    colores_gotico = [
        (0.02, 0.0, 0.03),   # negro violeta profundo
        (0.15, 0.0, 0.2),    # púrpura oscuro
        (0.3, 0.0, 0.25),    # vino oscuro
        (0.5, 0.0, 0.0),     # rojo sangre
        (0.7, 0.1, 0.1),     # rojo medio
        (0.9, 0.2, 0.3),     # rojo carmesí
        (1.0, 0.3, 0.4)      # rosado oscuro final
    ]
    
    # Crear colormap con muchos niveles (N=256) para suavizado
    return LinearSegmentedColormap.from_list("GothicMap", colores_gotico, N=256)

def apply_matplotlib_colormap(gray, cmap):
    g = gray.astype(np.float32) / 255.0
    rgb = cmap(g)[:, :, :3]
    rgb_255 = (rgb * 255).astype(np.uint8)
    bgr = rgb_255[:, :, ::-1]  # RGB->BGR
    return bgr

def bgr_to_png_bytes(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def gray_to_png_bytes(gray):
    pil_img = Image.fromarray(gray)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def tune_and_mix(bgr, gray, sat=1.0, val=1.0, mix=1.0):
    """Ajusta Saturación/Valor (HSV) y mezcla con grises.
    sat>1 aumenta saturación; val<1 oscurece; mix: 1=color, 0=gris."""
    # a RGB para HSV en OpenCV
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * float(sat), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * float(val), 0, 255)
    rgb2 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    bgr2 = cv2.cvtColor(rgb2, cv2.COLOR_RGB2BGR)

    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(bgr2, float(mix), gray_bgr, 1.0 - float(mix), 0)
    return out

# -------------- Sidebar (controles) --------------
st.sidebar.title("Controles")
file = st.sidebar.file_uploader("Cargar imagen (cualquier formato)", type=None)
resize_w = st.sidebar.slider("Ancho de trabajo (px)", 256, 2048, 768, step=64)
gamma = st.sidebar.slider("Gamma (pre-procesado)", 0.20, 3.00, 1.00, step=0.05)
show_gray = st.sidebar.checkbox("Mostrar imagen en grises", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Colormaps de OpenCV")
cv_maps_def = {
    "OpenCV • JET":   cv2.COLORMAP_JET,
    "OpenCV • HOT":   cv2.COLORMAP_HOT,
    "OpenCV • OCEAN": cv2.COLORMAP_OCEAN,
    "OpenCV • BONE":  cv2.COLORMAP_BONE,
    "OpenCV • PINK":  cv2.COLORMAP_PINK,
    "OpenCV • PARULA": cv2.COLORMAP_PARULA if hasattr(cv2, "COLORMAP_PARULA") else cv2.COLORMAP_TURBO,
    "OpenCV • TURBO": cv2.COLORMAP_TURBO,
}
sel_cv = st.sidebar.multiselect(
    "Selecciona mapas",
    options=list(cv_maps_def.keys()),
    default=["OpenCV • JET", "OpenCV • TURBO", "OpenCV • BONE"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Mapa Personalizado • Pastel")
use_pastel = st.sidebar.checkbox("Incluir Pastel personalizado", value=True)
pastel_brightness = st.sidebar.slider("Pastel • Brightness", 0.4, 1.2, 0.80, step=0.05)
sat_control = st.sidebar.slider("Saturación (HSV)", 0.50, 1.80, 1.15, step=0.05)
val_control = st.sidebar.slider("Valor / Brillo (HSV)", 0.50, 1.20, 0.85, step=0.05)
mix_control = st.sidebar.slider("Mezcla con grises", 0.0, 1.0, 0.90, step=0.05,
                                help="1.0 = solo color; 0.0 = solo grises")

st.sidebar.markdown("---")
st.sidebar.caption("Mapa Personalizado • Gótico 🦇")
use_gothic = st.sidebar.checkbox("Incluir Gótico personalizado", value=True)

st.sidebar.markdown("---")
grid_cols = st.sidebar.slider("Columnas en la rejilla", 2, 5, 3)
st.sidebar.caption("La rejilla se adapta a pantalla (layout responsivo).")

# -------------- Header --------------
st.title("Pseudocolor UI — Práctica 4")

# Descripción educativa
with st.expander("📚 ¿Qué es el Pseudocolor?", expanded=False):
    st.markdown("""
    El **pseudocolor** asigna colores artificiales a imágenes en escala de grises para:
    
    - 🎨 **Mejorar la visualización**: Los humanos distinguimos mejor colores que tonos de gris
    - 🔍 **Destacar características**: Diferentes rangos de intensidad se asignan a colores únicos
    - 🌡️ **Representar datos**: Común en imágenes térmicas, médicas y científicas
    
    **Mapas de color (Colormaps)**:
    - Cada valor de gris (0-255) se mapea a un color RGB específico
    - Permiten resaltar detalles que serían difíciles de ver en escala de grises
    
    **Controles disponibles**:
    - Gamma, saturación y brillo para ajustar la apariencia
    - Múltiples mapas predefinidos de OpenCV
    - Mapas personalizados (Pastel, Gótico)
    """)

st.write("Sube una imagen, elige colormaps y ajusta los parámetros para obtener visualizaciones impactantes.")


# -------------- Carga / Preproceso --------------
gray = to_gray_array(file, width=resize_w).copy()

# Corrección gamma previa (oscurece si gamma<1, aclara si >1)
if gamma != 1.0:
    g = np.power(gray.astype(np.float32) / 255.0, 1.0 / gamma)
    gray = np.clip(g * 255.0, 0, 255).astype(np.uint8)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# -------------- Procesamiento --------------
resultados = {}  # nombre -> BGR

# OpenCV maps
for name in sel_cv:
    flag = cv_maps_def[name]
    bgr = apply_opencv_colormap(gray, flag)
    resultados[name] = bgr

# Pastel personalizado (con brightness + HSV + mix)
if use_pastel:
    pastel = pastel_colormap(brightness=pastel_brightness)
    pastel_bgr = apply_matplotlib_colormap(gray, pastel)
    pastel_bgr = tune_and_mix(
        pastel_bgr, gray,
        sat=sat_control,
        val=val_control,
        mix=mix_control
    )
    resultados["Personalizado • Pastel"] = pastel_bgr

# Gótico personalizado (oscuro y dramático)
if use_gothic:
    gothic = gothic_colormap()
    gothic_bgr = apply_matplotlib_colormap(gray, gothic)
    # El gótico se ve mejor con menos saturación y un poco más oscuro
    gothic_bgr = tune_and_mix(
        gothic_bgr, gray,
        sat=0.95,  # Menos saturado para mantener el tono oscuro
        val=0.75,  # Valor más bajo para oscurecer
        mix=0.95   # Casi puro color
    )
    resultados["Personalizado • Gótico 🦇"] = gothic_bgr

# -------------- Vista previa --------------
with st.container():
    cols = st.columns(grid_cols, gap="large")

    # Imagen en grises
    if show_gray:
        with cols[0]:
            st.subheader("Escala de grises")
            st.image(gray, clamp=True, use_container_width=True)
            st.download_button(
                "Descargar (PNG)",
                data=gray_to_png_bytes(gray),
                file_name=f"EscalaGrises_{ts}.png",
                mime="image/png",
                use_container_width=True
            )

    # Resultados en grilla
    names = list(resultados.keys())
    for i, name in enumerate(names):
        col = cols[(i + (1 if show_gray else 0)) % grid_cols]
        with col:
            st.subheader(name)
            rgb = cv2.cvtColor(resultados[name], cv2.COLOR_BGR2RGB)
            st.image(rgb, use_container_width=True)
            st.download_button(
                "Descargar (PNG)",
                data=bgr_to_png_bytes(resultados[name]),
                file_name=f"{name.replace(' • ', '_').replace(' ', '')}_{ts}.png",
                mime="image/png",
                use_container_width=True
            )

# -------------- Comparativa y ZIP (con padding por fila) --------------
with st.container():
    st.markdown("### Exportar todo")

    thumbs = []
    labels = []

    if show_gray:
        # gris -> BGR -> RGB para uniformar a 3 canales
        thumbs.append(cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB))
        labels.append("Grises")

    for k in resultados.keys():
        thumbs.append(cv2.cvtColor(resultados[k], cv2.COLOR_BGR2RGB))
        labels.append(k)

    buf_comp = None  # por si no se llega a construir

    if len(thumbs) > 0:
        # 1) Igualamos alturas y reescalamos manteniendo aspecto
        max_h = max(im.shape[0] for im in thumbs)
        scaled = [
            cv2.resize(im, (int(im.shape[1] * (max_h / im.shape[0])), max_h), interpolation=cv2.INTER_CUBIC)
            for im in thumbs
        ]

        # 2) Construimos filas con separadores
        sep = 10
        spacer_col = 240  # gris claro
        spacer_v = np.ones((max_h, sep, 3), dtype=np.uint8) * spacer_col

        per_row = min(grid_cols, 4)  # no más de 4 por fila en comparativa
        row_imgs = []
        for i in range(0, len(scaled), per_row):
            row_parts = []
            for j, im in enumerate(scaled[i:i+per_row]):
                if j > 0:
                    row_parts.append(spacer_v)
                row_parts.append(im)
            row_img = np.concatenate(row_parts, axis=1)
            row_imgs.append(row_img)

        # 3) Pad a la misma anchura para poder concatenar en vertical
        max_w = max(r.shape[1] for r in row_imgs)
        padded_rows = []
        for r in row_imgs:
            if r.shape[1] < max_w:
                pad = np.ones((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8) * spacer_col
                r = np.concatenate([r, pad], axis=1)
            padded_rows.append(r)

        # 4) Concatenamos filas en vertical
        comp = padded_rows[0] if len(padded_rows) == 1 else np.concatenate(padded_rows, axis=0)

        # (Opcional) Banda superior
        banner_h = 40
        banner = np.ones((banner_h, comp.shape[1], 3), dtype=np.uint8) * 20
        comp_lab = np.concatenate([banner, comp], axis=0)

        # Mostrar y preparar descarga
        st.image(comp_lab, caption="Comparativa", use_container_width=True)
        comp_pil = Image.fromarray(comp_lab)
        buf_comp = io.BytesIO()
        comp_pil.save(buf_comp, format="PNG")
        buf_comp.seek(0)
        st.download_button(
            "Descargar comparativa (PNG)",
            data=buf_comp,
            file_name=f"Comparativa_{ts}.png",
            mime="image/png",
            use_container_width=True
        )

    # 5) ZIP con todo
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if show_gray:
            zf.writestr(f"EscalaGrises_{ts}.png", gray_to_png_bytes(gray).getvalue())
        for k, bgr in resultados.items():
            zf.writestr(
                f"{k.replace(' • ', '_').replace(' ', '')}_{ts}.png",
                bgr_to_png_bytes(bgr).getvalue()
            )
        if buf_comp is not None:
            zf.writestr(f"Comparativa_{ts}.png", buf_comp.getvalue())
    zip_buf.seek(0)

    st.download_button(
        "Descargar TODO (.zip)",
        data=zip_buf,
        file_name=f"Pseudocolor_Resultados_{ts}.zip",
        mime="application/zip",
        use_container_width=True
    )

# -------------- Footer --------------
with st.expander("Notas y tips"):
    st.markdown("""
- **Gamma** (pre) modifica la luminancia de entrada antes del pseudocolor.
- **Pastel • Brightness** (<1 oscurece) evita que el pastel quede blanco.
- **Saturación / Valor** ajustan el color en HSV; **Valor** <1 lo hace más opaco.
- **Mezcla con grises** combina color con la imagen base para darle más "cuerpo".
""")

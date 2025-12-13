# -*- coding: utf-8 -*-
"""
Práctica 5: Procesamiento en el Dominio de la Frecuencia
FFT (Filtrado) y DCT (Compresión tipo JPEG)
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# ========= Configuración de página =========
st.set_page_config(
    page_title="Procesamiento Frecuencia",
    page_icon="📊",
    layout="wide"
)

# ========= Funciones de FFT =========
def cargar_imagen_gray(uploaded_file, size=512):
    """Carga imagen en escala de grises y redimensiona."""
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('L')
        img = img.resize((size, size), Image.BICUBIC)
        return np.array(img, dtype=np.float32)
    # Imagen sintética (patrón de prueba)
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    img = 128 + 127 * np.sin(np.sqrt(X**2 + Y**2))
    return img.astype(np.float32)

def filtro_ideal(shape, cutoff, tipo='lowpass'):
    """Filtro ideal (paso bajo o alto)."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - ccol)**2 + (y - crow)**2)
    mask = np.zeros(shape, dtype=np.float32)
    if tipo == 'lowpass':
        mask[dist <= cutoff * min(rows, cols) / 2] = 1.0
    else:  # highpass
        mask[dist > cutoff * min(rows, cols) / 2] = 1.0
    return mask

def filtro_gaussiano(shape, cutoff, tipo='lowpass'):
    """Filtro gaussiano."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - ccol)**2 + (y - crow)**2)
    sigma = cutoff * min(rows, cols) / 2
    if tipo == 'lowpass':
        mask = np.exp(-(dist**2) / (2 * sigma**2))
    else:  # highpass
        mask = 1.0 - np.exp(-(dist**2) / (2 * sigma**2))
    return mask.astype(np.float32)

def filtro_butterworth(shape, cutoff, orden=2, tipo='lowpass'):
    """Filtro Butterworth."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - ccol)**2 + (y - crow)**2)
    D0 = cutoff * min(rows, cols) / 2
    epsilon = 1e-10
    if tipo == 'lowpass':
        mask = 1.0 / (1.0 + (dist / (D0 + epsilon))**(2 * orden))
    else:  # highpass
        mask = 1.0 / (1.0 + ((D0 + epsilon) / (dist + epsilon))**(2 * orden))
    return mask.astype(np.float32)

def normalizar_imagen(img):
    """Normaliza imagen a rango [0, 255] uint8."""
    img_norm = img.copy()
    img_norm = img_norm - img_norm.min()
    if img_norm.max() > 0:
        img_norm = img_norm / img_norm.max() * 255.0
    return np.clip(img_norm, 0, 255).astype(np.uint8)

def aplicar_filtro_fft(img, filtro_tipo, cutoff, orden=2, tipo_filtro='lowpass'):
    """Aplica filtro en dominio de frecuencia."""
    # FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    # Crear máscara
    if filtro_tipo == 'ideal':
        mask = filtro_ideal(img.shape, cutoff, tipo_filtro)
    elif filtro_tipo == 'gaussiano':
        mask = filtro_gaussiano(img.shape, cutoff, tipo_filtro)
    else:  # butterworth
        mask = filtro_butterworth(img.shape, cutoff, orden, tipo_filtro)
    
    # Aplicar filtro
    fshift_filtrado = fshift * mask
    
    # IFFT
    f_ishift = np.fft.ifftshift(fshift_filtrado)
    img_filtrada = np.fft.ifft2(f_ishift)
    img_filtrada = np.real(img_filtrada)
    
    # Normalizar a [0, 255]
    img_filtrada = normalizar_imagen(img_filtrada)
    
    # Espectro para visualización
    espectro = 20 * np.log10(np.abs(fshift) + 1)
    espectro_filtrado = 20 * np.log10(np.abs(fshift_filtrado) + 1)
    
    return img_filtrada, espectro, espectro_filtrado, mask

# ========= Funciones de DCT =========
def dct2(block):
    """DCT 2D de un bloque."""
    return np.dot(np.dot(dct_matrix(block.shape[0]), block), dct_matrix(block.shape[0]).T)

def idct2(block):
    """IDCT 2D de un bloque."""
    return np.dot(np.dot(dct_matrix(block.shape[0]).T, block), dct_matrix(block.shape[0]))

def dct_matrix(N):
    """Matriz DCT."""
    matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == 0:
                matrix[i, j] = np.sqrt(1.0 / N)
            else:
                matrix[i, j] = np.sqrt(2.0 / N) * np.cos((2 * j + 1) * i * np.pi / (2 * N))
    return matrix

# Matriz de cuantización JPEG estándar (luminancia)
QUANT_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

def comprimir_dct(img, q_factor=0.5):
    """Compresión DCT por bloques 8x8."""
    h, w = img.shape
    # Asegurar múltiplo de 8
    h_pad = ((h + 7) // 8) * 8
    w_pad = ((w + 7) // 8) * 8
    img_pad = np.zeros((h_pad, w_pad))
    img_pad[:h, :w] = img
    
    # Matriz de cuantización escalada
    Q = QUANT_MATRIX * (1.0 / q_factor) if q_factor > 0 else QUANT_MATRIX
    
    # Procesar por bloques
    img_dct = np.zeros_like(img_pad)
    for i in range(0, h_pad, 8):
        for j in range(0, w_pad, 8):
            block = img_pad[i:i+8, j:j+8]
            # DCT
            dct_block = dct2(block)
            # Cuantizar
            quant_block = np.round(dct_block / Q)
            # Descuantizar
            dequant_block = quant_block * Q
            # IDCT
            idct_block = idct2(dequant_block)
            img_dct[i:i+8, j:j+8] = idct_block
    
    return img_dct[:h, :w]

def calcular_psnr(original, reconstruida):
    """Calcula PSNR entre dos imágenes."""
    mse = np.mean((original - reconstruida) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# ========= Interfaz de Streamlit =========
st.title("📊 Procesamiento en el Dominio de la Frecuencia")
st.markdown("### FFT (Filtrado) y DCT (Compresión tipo JPEG)")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    
    uploaded_file = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is None:
        st.info("💡 Sin imagen = patrón sintético")
    
    img_size = st.slider("Tamaño de imagen", 256, 1024, 512, step=64)
    
    st.markdown("---")
    modo = st.radio("Seleccionar modo", ["Parte A: FFT y Filtrado", "Parte B: DCT y Compresión"])

# Cargar imagen
img_original = cargar_imagen_gray(uploaded_file, img_size)

# ========= PARTE A: FFT Y FILTRADO =========
if modo == "Parte A: FFT y Filtrado":
    st.subheader("🌊 Transformada de Fourier y Filtrado")
    
    with st.sidebar:
        st.markdown("---")
        st.caption("**Parámetros del Filtro**")
        
        filtro_tipo = st.selectbox("Tipo de filtro", ["ideal", "gaussiano", "butterworth"])
        tipo_filtro = st.radio("Modo", ["lowpass", "highpass"])
        cutoff = st.slider("Cutoff (radio normalizado)", 0.01, 0.50, 0.15, step=0.01)
        
        orden = 2
        if filtro_tipo == "butterworth":
            orden = st.slider("Orden (solo Butterworth)", 1, 5, 2)
        
        if st.button("🔄 Aplicar Filtro", use_container_width=True):
            st.session_state.aplicar_fft = True
    
    if st.session_state.get('aplicar_fft', False):
        with st.spinner('Procesando FFT...'):
            img_filtrada, espectro, espectro_filtrado, mask = aplicar_filtro_fft(
                img_original, filtro_tipo, cutoff, orden, tipo_filtro
            )
            
            st.session_state.img_filtrada = img_filtrada
            st.session_state.espectro = espectro
            st.session_state.espectro_filtrado = espectro_filtrado
            st.session_state.mask = mask
    
    # Visualización
    if 'img_filtrada' in st.session_state:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Imagen Original**")
            # Convertir a uint8 para visualización
            img_orig_display = normalizar_imagen(img_original)
            st.image(img_orig_display, clamp=True, use_container_width=True, channels="GRAY")
            
            st.markdown("**Espectro de Frecuencia (Original)**")
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            ax1.imshow(st.session_state.espectro, cmap='gray')
            ax1.set_title('Espectro |F(u,v)|')
            ax1.axis('off')
            st.pyplot(fig1)
            plt.close()
        
        with col2:
            st.markdown("**Imagen Filtrada**")
            st.image(st.session_state.img_filtrada, clamp=True, use_container_width=True, channels="GRAY")
            
            st.markdown("**Espectro Filtrado**")
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.imshow(st.session_state.espectro_filtrado, cmap='gray')
            ax2.set_title('Espectro Filtrado')
            ax2.axis('off')
            st.pyplot(fig2)
            plt.close()
        
        # Máscara del filtro
        st.markdown("---")
        st.markdown("**Máscara del Filtro**")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m2:
            fig3, ax3 = plt.subplots(figsize=(5, 5))
            ax3.imshow(st.session_state.mask, cmap='gray')
            ax3.set_title(f'{filtro_tipo.capitalize()} {tipo_filtro} (cutoff={cutoff})')
            ax3.axis('off')
            st.pyplot(fig3)
            plt.close()
        
        # Info técnica
        with st.expander("ℹ️ Información del Filtrado"):
            st.markdown(f"""
            - **Filtro**: {filtro_tipo.capitalize()}
            - **Tipo**: {tipo_filtro.upper()}
            - **Cutoff**: {cutoff}
            - **Orden**: {orden if filtro_tipo == 'butterworth' else 'N/A'}
            - **Tamaño**: {img_original.shape[0]}x{img_original.shape[1]}
            
            **Interpretación**:
            - **Lowpass**: Deja pasar bajas frecuencias (suaviza, elimina ruido)
            - **Highpass**: Deja pasar altas frecuencias (resalta bordes)
            - **Cutoff**: Radio de corte normalizado (0-0.5)
            """)
    else:
        st.info("👈 Configura los parámetros y presiona **Aplicar Filtro**")

# ========= PARTE B: DCT Y COMPRESIÓN =========
else:  # Parte B
    st.subheader("🗜️ Compresión DCT (tipo JPEG)")
    
    with st.sidebar:
        st.markdown("---")
        st.caption("**Parámetros de Compresión**")
        
        q_factor = st.slider("Factor de Calidad (q_factor)", 0.1, 2.0, 0.5, step=0.1,
                            help="Menor valor = mayor compresión (más pérdida)")
        
        if st.button("🗜️ Comprimir con DCT", use_container_width=True):
            st.session_state.aplicar_dct = True
    
    if st.session_state.get('aplicar_dct', False):
        with st.spinner('Procesando DCT...'):
            img_comprimida = comprimir_dct(img_original, q_factor)
            psnr = calcular_psnr(img_original, img_comprimida)
            
            st.session_state.img_comprimida = img_comprimida
            st.session_state.psnr = psnr
            st.session_state.q_factor_usado = q_factor
    
    # Visualización
    if 'img_comprimida' in st.session_state:
        # Métricas
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Factor de Calidad", f"{st.session_state.q_factor_usado:.2f}")
        with col_m2:
            st.metric("PSNR", f"{st.session_state.psnr:.2f} dB")
        with col_m3:
            calidad = "Excelente" if st.session_state.psnr > 40 else \
                      "Buena" if st.session_state.psnr > 30 else \
                      "Aceptable" if st.session_state.psnr > 25 else "Baja"
            st.metric("Calidad Perceptual", calidad)
        
        st.markdown("---")
        
        # Comparación lado a lado
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Imagen Original**")
            img_orig_display = normalizar_imagen(img_original)
            st.image(img_orig_display, clamp=True, use_container_width=True, channels="GRAY")
        
        with col2:
            st.markdown("**Imagen Comprimida (DCT)**")
            img_comp_display = normalizar_imagen(st.session_state.img_comprimida)
            st.image(img_comp_display, clamp=True, use_container_width=True, channels="GRAY")
        
        # Diferencia
        st.markdown("---")
        st.markdown("**Mapa de Error (diferencia amplificada)**")
        diferencia = np.abs(img_original - st.session_state.img_comprimida)
        col_d1, col_d2, col_d3 = st.columns([1, 2, 1])
        with col_d2:
            fig_diff, ax_diff = plt.subplots(figsize=(8, 8))
            im = ax_diff.imshow(diferencia * 10, cmap='hot')
            ax_diff.set_title('Error × 10')
            ax_diff.axis('off')
            plt.colorbar(im, ax=ax_diff, fraction=0.046)
            st.pyplot(fig_diff)
            plt.close()
        
        # Información técnica
        with st.expander("ℹ️ Sobre la Compresión DCT"):
            st.markdown(f"""
            **DCT (Discrete Cosine Transform)**:
            - Divide la imagen en bloques de **8×8 píxeles**
            - Aplica DCT 2D a cada bloque
            - **Cuantiza** los coeficientes (descarta información menos importante)
            - Reconstruye con IDCT
            
            **Parámetros**:
            - **q_factor = {st.session_state.q_factor_usado:.2f}**
            - Factor < 1: Mayor compresión, menor calidad
            - Factor > 1: Menor compresión, mayor calidad
            
            **PSNR (Peak Signal-to-Noise Ratio)**:
            - Métrica objetiva de calidad
            - **{st.session_state.psnr:.2f} dB** ← Valor actual
            - > 40 dB: Excelente
            - 30-40 dB: Buena
            - 25-30 dB: Aceptable
            - < 25 dB: Baja calidad
            
            **Uso**: Similar a compresión JPEG estándar
            """)
        
        # Comparación de calidades
        st.markdown("---")
        st.markdown("### 📊 Comparación de Calidades")
        
        if st.button("Generar comparación (q=0.3, 0.6, 1.0)"):
            with st.spinner("Generando comparación..."):
                q_values = [0.3, 0.6, 1.0]
                cols_comp = st.columns(len(q_values))
                
                for i, q in enumerate(q_values):
                    img_comp = comprimir_dct(img_original, q)
                    psnr_comp = calcular_psnr(img_original, img_comp)
                    
                    with cols_comp[i]:
                        st.markdown(f"**q = {q}**")
                        img_comp_display = normalizar_imagen(img_comp)
                        st.image(img_comp_display, clamp=True, use_container_width=True, channels="GRAY")
                        st.caption(f"PSNR: {psnr_comp:.2f} dB")
    else:
        st.info("👈 Ajusta el **q_factor** y presiona **Comprimir con DCT**")

# Footer
st.markdown("---")
with st.expander("📚 Objetivos de Aprendizaje"):
    st.markdown("""
    ### Parte A - FFT y Filtrado
    - Interpretar el **espectro de frecuencia** de una imagen
    - Reconocer la relación entre **contenido espacial** y **frecuencia**
    - Diseñar **máscaras de filtrado** (Ideal, Gaussiano, Butterworth)
    - Comprender **lowpass** (suavizado) vs **highpass** (realce de bordes)
    - Practicar **reconstrucción por IFFT**
    
    ### Parte B - DCT y Compresión
    - Entender la **transformada del coseno discreta** (DCT)
    - Comprender la **cuantización** en compresión JPEG
    - Analizar el compromiso **calidad/compresión** con q_factor
    - Usar métricas objetivas (**PSNR**) para evaluar pérdida
    - Interpretar **pérdida perceptual** en imágenes comprimidas
    """)

st.caption("Procesamiento en Dominio de Frecuencia | Image Analysis 2025")

# Proyecto Image Analysis

Colección de proyectos desarrollados durante el semestre en el curso de Image Analysis.

## 🚀 Instalación

1. Clona este repositorio
2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## 📖 Uso

Para ejecutar la aplicación:

```bash
streamlit run Home.py
```

## 📁 Estructura del Proyecto

```
proyecto_image_analysis/
│
├── Home.py                 # Página principal
├── pages/                  # Páginas adicionales (módulos del semestre)
│   ├── 01_Procesamiento_de_Imagenes.py
│   ├── 02_Operaciones_sobre_Imagenes.py
│   └── ...
├── requirements.txt        # Dependencias del proyecto
├── .gitignore             # Archivos ignorados por Git
└── README.md              # Este archivo
```

## ➕ Agregar Nuevos Módulos

Para agregar un nuevo programa al proyecto:

1. Crea un archivo `.py` en la carpeta `pages/`
2. Nómbralo con el formato: `03_Nombre_del_Modulo.py`
3. El número al inicio define el orden en el menú lateral
4. El archivo aparecerá automáticamente en el menú de navegación

Ejemplo:
```python
import streamlit as st

st.title("Mi Nuevo Módulo")
st.write("Contenido del módulo...")
```

## 📚 Módulos Incluidos

### 01 - Procesamiento de Imágenes
- Conversión a múltiples modelos de color (RGB, YIQ, CMY, HSV)
- Separación de canales RGB con realce de color
- Escala de grises (BT.601)
- Binarización con umbral ajustable
- Visualización de histogramas

### 02 - Operaciones sobre Imágenes
- **Operaciones Aritméticas**: Suma, resta, multiplicación, lightest, darkest
- **Operaciones Lógicas**: AND, OR, XOR, NOT
- **Operaciones Relacionales**: A > B, A < B, A == B
- **Componentes Conexas**: Análisis con conectividad 4 u 8

## 🛠️ Requisitos

- Python 3.12+
- Streamlit
- OpenCV (opencv-python)
- NumPy
- Pandas
- Matplotlib
- Pillow

---

**Semestre 2025** - Image Analysis

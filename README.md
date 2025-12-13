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
│   └── ...
├── requirements.txt        # Dependencias del proyecto
└── README.md              # Este archivo
```

## ➕ Agregar Nuevos Módulos

Para agregar un nuevo programa al proyecto:

1. Crea un archivo `.py` en la carpeta `pages/`
2. Nómbralo con el formato: `01_Nombre_del_Modulo.py`
3. El número al inicio define el orden en el menú lateral
4. El archivo aparecerá automáticamente en el menú de navegación

Ejemplo:
```python
import streamlit as st

st.title("Mi Nuevo Módulo")
st.write("Contenido del módulo...")
```

## 📚 Módulos Incluidos

_Los módulos se irán agregando conforme se complete el semestre_

---

**Semestre 2025** - Image Analysis

"""
Módulo de Clasificación de Calidad de Frutas.
Integrado en la plataforma Spectra.
"""

import os
import sys
from pathlib import Path
import streamlit as st

# Configurar el path para encontrar los módulos de fruit_quality_project
PROJECT_ROOT = Path(__file__).parent.parent
FRUIT_PROJECT_DIR = PROJECT_ROOT / "fruit_quality_project"

# Agregar al path para que las importaciones internas de app.py funcionen
if str(FRUIT_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(FRUIT_PROJECT_DIR))

# Importar la lógica principal de la aplicación de frutas
# Importamos main como a_main para evitar conflictos si hubiera
try:
    from app import main as fruit_main
except ImportError as e:
    st.error(f"No se pudo cargar el módulo de frutas: {e}")
    st.info(f"Buscando en: {FRUIT_PROJECT_DIR}")
    fruit_main = None

if __name__ == "__main__":
    if fruit_main:
        fruit_main()
    else:
        st.error("La aplicación de clasificación de frutas no está disponible.")

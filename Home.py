import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Proyecto Image Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🔬 Proyecto de Image Analysis")
st.markdown("---")

# Descripción del proyecto
st.markdown("""
## Bienvenido

Este es el repositorio de proyectos desarrollados durante el semestre en el curso de **Image Analysis**.

### 📚 Acerca de este proyecto

En esta aplicación encontrarás los diferentes programas y herramientas desarrollados 
a lo largo del semestre, organizados en módulos independientes.

### 🧭 Navegación

Utiliza el **menú lateral** (sidebar) para acceder a los diferentes programas y módulos 
que hemos desarrollado.

### 🎯 Instrucciones

1. Selecciona un módulo del menú lateral
2. Cada módulo contiene su propia funcionalidad independiente
3. Sigue las instrucciones específicas de cada programa

---

""")

# Información adicional
col1, col2 = st.columns(2)

with col1:
    st.info("💡 **Tip**: Cada módulo es independiente y puede ejecutarse por separado.")

with col2:
    st.success("✅ Explora los diferentes programas en el menú lateral")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Proyecto Image Analysis - Semestre 2025</p>
</div>
""", unsafe_allow_html=True)

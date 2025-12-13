"""
Estilos compartidos para todas las páginas del proyecto
"""
import streamlit as st

def aplicar_estilo_morado():
    """Aplica el tema morado con degradado a la página."""
    st.markdown("""
    <style>
        /* Fondo principal con degradado morado */
        .stApp {
            background: linear-gradient(135deg, #1a0b2e 0%, #2d1b4e 50%, #16213e 100%);
        }
        
        /* Sidebar con degradado morado oscuro */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f0324 0%, #1a0b2e 100%);
        }
        
        /* Textos en color claro para buen contraste */
        .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, label {
            color: #ede9fe !important;
        }
        
        /* Tarjetas info/success/warning con fondo semi-transparente */
        .stAlert {
            background-color: rgba(45, 27, 78, 0.6) !important;
            border-left-color: #9f7aea !important;
        }
        
        /* Botones con estilo morado */
        .stButton > button {
            background: linear-gradient(90deg, #6b46c1 0%, #9f7aea 100%);
            color: white;
            border: none;
            border-radius: 8px;
        }
        
        .stButton > button:hover {
            background: linear-gradient(90deg, #553c9a 0%, #805ad5 100%);
        }
        
        /* Tarjetas de métricas */
        [data-testid="stMetricValue"] {
            color: #d8b4fe !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: rgba(15, 3, 36, 0.4);
            border-radius: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #c4b5fd;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #6b46c1 !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

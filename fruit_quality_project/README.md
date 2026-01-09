# 🍎 Fruit Quality Classification Project

Sistema completo para **segmentación de frutas** y **clasificación de calidad** (fresh vs rotten) utilizando técnicas clásicas de visión por computadora y deep learning con PyTorch.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Descarga del Dataset](#-descarga-del-dataset)
- [Ejecución](#-ejecución)
- [Métodos Utilizados](#-métodos-utilizados)
- [Resultados](#-resultados)

---

## 📖 Descripción

Este proyecto implementa un pipeline completo para:

1. **Segmentar frutas** usando técnicas clásicas (GrabCut, HSV+morfología)
2. **Clasificar la calidad** (fresh vs rotten) mediante transfer learning con MobileNetV2
3. **Comparar el rendimiento** con y sin segmentación
4. **Visualizar resultados** a través de una interfaz interactiva en Streamlit

### Características principales:
- ✅ Dos métodos de segmentación seleccionables
- ✅ Transfer learning con backbone congelado y fine-tuning
- ✅ Métricas completas (Accuracy, Precision, Recall, F1)
- ✅ Comparación experimental baseline vs segmentación
- ✅ Interfaz web interactiva con Streamlit

---

## 🎯 Demo Mode (Inference Only)

**No dataset required!** If you already have a trained model, you can run the Streamlit demo directly.

### Quick Start

```bash
cd fruit_quality_project

# Install dependencies
pip install -r requirements.txt

# Run the demo
streamlit run app.py
```

### Requirements for Demo Mode
- Trained model at: `models/fruit_quality_baseline.pth`
- No dataset needed - just upload images to classify

### CLI Inference (Optional)
```bash
python -m src.inference --image path/to/fruit.jpg --preprocess none
python -m src.inference --image path/to/fruit.jpg --preprocess grabcut
```

---

## 📁 Estructura del Proyecto

```
fruit_quality_project/
│
├── models/                    # Modelos entrenados (.pth) ⭐ Required for demo
│   └── fruit_quality_baseline.pth
├── data/                      # Dataset (only for training)
│   ├── train/
│   │   ├── freshapples/
│   │   ├── freshbananas/
│   │   ├── freshoranges/
│   │   ├── rottenapples/
│   │   ├── rottenbananas/
│   │   └── rottenoranges/
│   └── test/
│       └── (misma estructura)
│
├── outputs/
│   ├── segmentation_samples/  # Ejemplos de segmentación
│   ├── predictions_samples/   # Predicciones del modelo
│   └── streamlit_samples/     # Evaluaciones desde la app
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration and paths ⭐ NEW
│   ├── inference.py           # Standalone inference module ⭐ NEW
│   ├── segmentation.py        # GrabCut + HSV segmentation
│   ├── dataset.py             # PyTorch Dataset
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Metrics & evaluation
│   └── utils.py               # Utilities
│
├── main.py                    # Pipeline completo (training)
├── app.py                     # Streamlit app (demo)
├── requirements.txt
└── README.md
```

---

## 💻 Instalación

### 1. Clonar o descargar el proyecto

```bash
cd fruit_quality_project
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar instalación de PyTorch con CUDA (opcional)

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## 📥 Descarga del Dataset

### Opción A: Usando Kaggle API (recomendado)

1. **Instalar Kaggle API:**
   ```bash
   pip install kaggle
   ```

2. **Configurar credenciales:**
   - Ve a [kaggle.com/account](https://www.kaggle.com/account)
   - Click "Create New API Token"
   - Guarda `kaggle.json` en `~/.kaggle/` (Linux) o `C:\Users\<user>\.kaggle\` (Windows)

3. **Descargar dataset:**
   ```bash
   kaggle datasets download -d sriramr/fruits-fresh-and-rotten-for-classification
   ```

4. **Extraer en la carpeta data/:**
   ```bash
   unzip fruits-fresh-and-rotten-for-classification.zip -d data/
   ```

### Opción B: Descarga manual

1. Ve a [Kaggle Dataset](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
2. Click "Download"
3. Extrae el contenido en la carpeta `data/`

La estructura final debe ser:
```
data/
├── train/
│   ├── freshapples/
│   ├── freshbananas/
│   ├── freshoranges/
│   ├── rottenapples/
│   ├── rottenbananas/
│   └── rottenoranges/
└── test/
    └── (misma estructura)
```

---

## 🚀 Ejecución

### Entrenamiento completo (main.py)

```bash
# Entrenar ambos modelos (baseline y con segmentación)
python main.py

# Con parámetros personalizados
python main.py --epochs 20 --batch-size 32 --segmentation-method grabcut

# Modo de prueba rápida (2 epochs)
python main.py --test-mode

# Solo baseline (sin segmentación)
python main.py --skip-segmented

# Solo con segmentación
python main.py --skip-baseline
```

### Parámetros disponibles:

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--data-dir` | `./data` | Ruta al dataset |
| `--epochs` | `15` | Épocas de entrenamiento |
| `--batch-size` | `32` | Tamaño de batch |
| `--segmentation-method` | `grabcut` | `grabcut` o `hsv` |
| `--test-mode` | | Modo prueba (2 epochs) |

### Aplicación Streamlit

```bash
streamlit run app.py
```

Abre `http://localhost:8501` en tu navegador.

---

## 🔬 Métodos Utilizados

### Segmentación

#### 1. GrabCut (OpenCV)
```
Algoritmo iterativo de segmentación basado en grafos
- Inicialización: rectángulo automático (margen de 10px)
- Iteraciones: 5 (configurable)
- Output: máscara binaria, imagen segmentada, bounding box
```

#### 2. HSV + Morfología
```
Segmentación por umbralización de color en espacio HSV
- Detección automática de rangos de color
- Operaciones morfológicas: opening + closing
- Kernel elíptico de 5x5
- Selección del contorno más grande
```

### Clasificación

#### Modelo: MobileNetV2 (Transfer Learning)
```
- Pretrained: ImageNet (IMAGENET1K_V1)
- Backbone: Congelado inicialmente
- Fine-tuning: Después de epoch 5
- Classifier: Linear(1280, 2)
- Optimizer: Adam (lr=0.001, fine-tune: lr=0.0001)
- Scheduler: ReduceLROnPlateau
```

#### Preprocesamiento
```
- Resize: 224×224
- Normalización: ImageNet mean/std
- Augmentation (train): RandomCrop, HorizontalFlip, Rotation, ColorJitter
```

---

## 📊 Resultados

### Métricas esperadas (referencia)

| Experimento | Accuracy | F1 (Macro) | Precision | Recall |
|-------------|----------|------------|-----------|--------|
| Baseline | ~82-85% | ~82-85% | ~82-85% | ~82-85% |
| Con segmentación | ~84-88% | ~84-88% | ~84-88% | ~84-88% |

### Outputs generados

- `outputs/segmentation_samples/` - Ejemplos de segmentación por clase
- `outputs/baseline/confusion_matrix.png` - Matriz de confusión baseline
- `outputs/segmented/confusion_matrix.png` - Matriz de confusión con segmentación
- `outputs/experiment_comparison.txt` - Comparación de experimentos
- `models/fruit_quality_baseline.pth` - Modelo sin segmentación
- `models/fruit_quality_segmented.pth` - Modelo con segmentación

### Conclusión

La segmentación mejora el rendimiento cuando:
- El fondo de las imágenes es variable
- Hay ruido visual o iluminación inconsistente
- El objeto de interés ocupa una porción pequeña de la imagen

---

## 📚 Referencias

- Dataset: [Fruits fresh and rotten for classification (Kaggle)](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
- MobileNetV2: [Sandler et al., 2018](https://arxiv.org/abs/1801.04381)
- GrabCut: [Rother et al., 2004](https://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf)

---

## 📝 Licencia

Este proyecto es para fines educativos - Image Analysis Course 2026.

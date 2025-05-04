# Duck-Tracker Project

Sistema de seguimiento y análisis de trayectorias de patos utilizando visión por computadora y YOLO.

## 📊 [Ver Informe Interactivo](https://fluffy-flan-360713.netlify.app/)

## Descripción

Este proyecto implementa un sistema de seguimiento y análisis del movimiento de patos en video, utilizando YOLOv8 para la detección y filtros de Kalman para el seguimiento. El sistema incluye visualizaciones avanzadas de trayectorias, análisis estadístico del movimiento y generación de informes interactivos.

## Características principales

- 🦆 Detección y seguimiento de patos en video
- 🔍 Filtro de Kalman para predicción de movimiento
- 📈 Visualización de trayectorias 2D y 3D
- 🎬 Animaciones de seguimiento con colores vibrantes
- 📊 Análisis estadístico del movimiento
- 📱 Informe HTML interactivo con visualizaciones

## Estructura del proyecto

- `batch_processing.py`: Procesamiento por lotes de videos
- `create_animation.py`: Creación de animaciones de trayectorias
- `create_report.py`: Generación de informe HTML interactivo
- `duck_test1.py`: Clase principal del tracker con Kalman Filter
- `movement_analysis.py`: Análisis estadístico de movimiento
- `visualize_trajectories.py`: Visualización de trayectorias 2D y 3D
- `assets/`: Videos de muestra y recursos
- `batch_output/`: Resultados del procesamiento y visualizaciones
- `best.pt`: Modelo YOLOv8 entrenado

## Requisitos

```
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
tqdm>=4.62.0
ultralytics>=8.0.0
opencv-python>=4.5.0
pygments>=2.10.0
```

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/AldonDC/Duck-Tracker-Project.git
cd Duck-Tracker-Project
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

1. Procesar un video por lotes:
```bash
python batch_processing.py --video assets/DuckVideo.mp4 --output batch_output/
```

2. Generar visualizaciones:
```bash
python visualize_trajectories.py
```

3. Crear animaciones:
```bash
python create_animation.py
```

4. Generar informe HTML interactivo:
```bash
python create_report.py
```

5. Ver el informe en un navegador web:
   - Abrir `batch_output/informe/informe_avanzado_patos.html`
   - O visitar [la versión online en Netlify](https://fluffy-flan-360713.netlify.app/)

## Enlaces

- [Repositorio GitHub](https://github.com/AldonDC/Duck-Tracker-Project)
- [Informe interactivo](https://fluffy-flan-360713.netlify.app/)

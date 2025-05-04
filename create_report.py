import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

def create_enhanced_html_report(data_file, visualizations_folder, output_folder, 
                                code_files=None, animation_files=None):
    """
    Crea un informe HTML mejorado con todas las visualizaciones, análisis,
    código fuente y animaciones
    
    Args:
        data_file: Ruta al archivo JSON con los datos combinados
        visualizations_folder: Carpeta con las visualizaciones generadas
        output_folder: Carpeta donde guardar el informe
        code_files: Lista de archivos Python para incluir como código fuente
        animation_files: Lista de archivos de animación (MP4, GIF)
    """
    # Crear carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    # Valores por defecto
    if code_files is None:
        code_files = []
    
    if animation_files is None:
        animation_files = [
            os.path.join(visualizations_folder, 'animacion_trayectorias_lenta.mp4'),
            os.path.join(visualizations_folder, 'animacion_trayectorias_lenta.gif')
        ]
    
    # Cargar datos para obtener información general
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Extraer información general
    total_frames = len(data.get('frames', {}))
    first_frame_positions = data.get('frames', {}).get('0', {}).get('positions', {})
    total_ducks = len(first_frame_positions)
    
    # Función para calcular estadísticas adicionales
    def calculate_additional_stats(data):
        frames_data = data.get('frames', {})
        
        # Estructuras para almacenar datos por pato
        duck_data = {}
        
        # Procesar cada frame
        for frame_idx, frame_data in sorted(frames_data.items(), key=lambda x: int(x[0])):
            frame_number = int(frame_data.get('frame_number', frame_idx))
            positions = frame_data.get('positions', {})
            
            for duck_id, duck_info in positions.items():
                if duck_id not in duck_data:
                    duck_data[duck_id] = {
                        'positions': [],
                        'frames': [],
                        'color': duck_info.get('color', 'yellow')
                    }
                
                position = duck_info.get('position')
                if position:
                    duck_data[duck_id]['positions'].append(position)
                    duck_data[duck_id]['frames'].append(frame_number)
        
        # Calcular métricas adicionales
        stats = []
        for duck_id, info in duck_data.items():
            positions = np.array(info['positions'])
            frames = np.array(info['frames'])
            
            if len(positions) > 1:
                # Calcular distancias entre puntos consecutivos
                deltas = positions[1:] - positions[:-1]
                distances = np.sqrt(deltas[:, 0]**2 + deltas[:, 1]**2)
                total_distance = np.sum(distances)
                
                # Calcular velocidades (distancia / cambio de frame)
                frame_deltas = frames[1:] - frames[:-1]
                speeds = distances / np.maximum(frame_deltas, 1)  # Evitar división por cero
                avg_speed = np.mean(speeds)
                max_speed = np.max(speeds)
                
                # Calcular aceleración
                if len(speeds) > 1:
                    accelerations = speeds[1:] - speeds[:-1]
                    avg_acceleration = np.mean(np.abs(accelerations))
                    max_acceleration = np.max(np.abs(accelerations))
                else:
                    avg_acceleration = 0
                    max_acceleration = 0
                
                # Duración
                duration = frames[-1] - frames[0] + 1
                
                # Calcular dirección predominante
                if len(deltas) > 0:
                    angles = np.arctan2(deltas[:, 1], deltas[:, 0]) * 180 / np.pi
                    # Convertir a 8 direcciones cardinales
                    directions = np.round(angles / 45) % 8
                    dir_counts = np.bincount(directions.astype(int), minlength=8)
                    main_dir_idx = np.argmax(dir_counts)
                    dir_names = ['Este', 'Noreste', 'Norte', 'Noroeste', 
                                 'Oeste', 'Suroeste', 'Sur', 'Sureste']
                    main_direction = dir_names[main_dir_idx]
                    dir_percentage = dir_counts[main_dir_idx] / len(deltas) * 100
                else:
                    main_direction = "N/A"
                    dir_percentage = 0
                
                # Puntos inicial y final
                start_pos = positions[0]
                end_pos = positions[-1]
                
                # Distancia en línea recta entre inicio y fin
                direct_distance = np.sqrt((end_pos[0]-start_pos[0])**2 + (end_pos[1]-start_pos[1])**2)
                
                # Ratio de eficiencia (distancia directa / distancia total)
                efficiency = direct_distance / total_distance if total_distance > 0 else 0
                
                # Almacenar estadísticas
                stats.append({
                    'duck_id': duck_id,
                    'color': info['color'],
                    'num_frames': len(frames),
                    'total_distance': total_distance,
                    'average_speed': avg_speed,
                    'max_speed': max_speed,
                    'avg_acceleration': avg_acceleration,
                    'max_acceleration': max_acceleration,
                    'duration': duration,
                    'main_direction': main_direction,
                    'dir_percentage': dir_percentage,
                    'efficiency': efficiency,
                    'start_x': start_pos[0],
                    'start_y': start_pos[1],
                    'end_x': end_pos[0],
                    'end_y': end_pos[1]
                })
        
        return pd.DataFrame(stats) if stats else None
    
    # Calcular estadísticas detalladas
    stats_df = calculate_additional_stats(data)
    
    # Leer CSV de estadísticas si existe, de lo contrario usar las calculadas
    stats_file = os.path.join(visualizations_folder, 'estadisticas_patos.csv')
    if os.path.exists(stats_file):
        csv_stats_df = pd.read_csv(stats_file)
        # Combinar con estadísticas calculadas si es necesario
        if stats_df is not None:
            # Usar columnas de ambos dataframes sin duplicados
            stats_df = pd.merge(stats_df, csv_stats_df, on='duck_id', how='outer', suffixes=('', '_csv'))
    
    # Leer archivos de código fuente
    code_snippets = {}
    for code_file in code_files:
        if os.path.exists(code_file):
            with open(code_file, 'r') as f:
                code_content = f.read()
                code_snippets[os.path.basename(code_file)] = code_content
    
    # Lista extendida de imágenes de visualización con descripciones detalladas
    visualization_files = [
        ('trayectorias_3d_colores.png', 'Trayectorias 3D en Espacio-Tiempo', 
         'Visualización tridimensional de las trayectorias de los patos a lo largo del tiempo, mostrando ' + 
         'la evolución espacial y temporal del movimiento de cada individuo.'),
        
        ('trayectorias_2d_colores.png', 'Trayectorias 2D en el Plano', 
         'Proyección bidimensional de las trayectorias completas de los patos, permitiendo analizar ' + 
         'los patrones de movimiento y las áreas más transitadas en el espacio de la escena.'),
        
        ('mapa_densidad_trayectorias_colores.png', 'Mapa de Densidad con Trayectorias', 
         'Combinación de un mapa de calor que muestra la densidad de posiciones con las trayectorias ' + 
         'individuales superpuestas, revelando tanto el comportamiento individual como colectivo.'),
        
        ('distancia_total_por_pato.png', 'Distancia Total Recorrida por Pato', 
         'Comparativa de la distancia total recorrida por cada pato durante toda la grabación, ' + 
         'identificando los individuos más activos y los más sedentarios.'),
        
        ('velocidad_promedio_por_pato.png', 'Velocidad Promedio por Pato', 
         'Análisis comparativo de la velocidad promedio de desplazamiento de cada pato, ' + 
         'permitiendo identificar diferencias en la movilidad individual.'),
        
        ('velocidad_vs_distancia.png', 'Relación entre Velocidad y Distancia', 
         'Gráfico de dispersión que muestra la correlación entre la velocidad promedio y la ' + 
         'distancia total recorrida para cada pato, revelando patrones de comportamiento.'),
        
        ('duracion_por_pato.png', 'Tiempo de Aparición en Video', 
         'Duración total de la presencia de cada pato en la grabación, mostrando ' + 
         'cuáles permanecieron visibles durante más tiempo.'),
        
        ('evolucion_velocidad.png', 'Evolución Temporal de la Velocidad', 
         'Análisis de cómo cambia la velocidad de los patos a lo largo del tiempo, ' + 
         'revelando patrones de aceleración, desaceleración y posibles eventos de interés.'),
        
        ('mapa_calor_cuadricula.png', 'Mapa de Calor en Cuadrícula', 
         'Representación de la densidad de presencia de patos en una cuadrícula que divide ' + 
         'el espacio de la escena, identificando zonas de alta concentración.'),
        
        ('direcciones_movimiento.png', 'Distribución de Direcciones', 
         'Análisis de las direcciones predominantes de movimiento, mostrando si hay ' + 
         'tendencias direccionales específicas en el comportamiento de los patos.'),
        
        ('correlacion_metricas.png', 'Correlación entre Métricas de Movimiento', 
         'Matriz de correlación entre diferentes métricas de movimiento, revelando ' + 
         'relaciones entre variables como velocidad, aceleración, distancia y duración.')
    ]
    
    # Función para destacar código Python
    def format_python_code(code):
        formatter = HtmlFormatter(style='monokai', linenos=True, cssclass='codehilite')
        highlighted = highlight(code, PythonLexer(), formatter)
        css = formatter.get_style_defs('.codehilite')
        return highlighted, css
    
    # Obtener CSS para el código destacado
    code_css = ""
    if code_snippets:
        _, code_css = format_python_code(next(iter(code_snippets.values())))
    
    # Preparar contenido HTML del informe
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Análisis Avanzado de Trayectorias de Patos</title>
        <style>
            :root {{
                --primary-color: #3498db;
                --secondary-color: #2c3e50;
                --accent-color: #e74c3c;
                --light-bg: #f9f9f9;
                --dark-bg: #34495e;
                --text-color: #333;
                --light-text: #ecf0f1;
                --border-radius: 8px;
                --box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: var(--text-color);
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: var(--light-bg);
            }}
            
            h1, h2, h3, h4 {{
                color: var(--secondary-color);
                margin-top: 1.5em;
            }}
            
            h1 {{
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 2px solid var(--primary-color);
                padding-bottom: 15px;
                font-size: 2.5em;
            }}
            
            h2 {{
                border-left: 5px solid var(--primary-color);
                padding-left: 15px;
                font-size: 1.8em;
                margin-top: 2em;
            }}
            
            h3 {{
                font-size: 1.4em;
                border-bottom: 1px dashed var(--primary-color);
                padding-bottom: 5px;
            }}
            
            /* Tarjetas para visualizaciones */
            .visualization-card {{
                margin: 30px 0;
                background: white;
                border-radius: var(--border-radius);
                box-shadow: var(--box-shadow);
                padding: 20px;
                transition: transform 0.3s ease;
            }}
            
            .visualization-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            }}
            
            .visualization-card img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 0 auto;
                border-radius: 5px;
            }}
            
            .card-caption {{
                text-align: center;
                margin-top: 15px;
                font-weight: bold;
                color: var(--secondary-color);
            }}
            
            .card-description {{
                text-align: justify;
                margin-top: 10px;
                color: #555;
            }}
            
            /* Tabla de estadísticas */
            .stats-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                box-shadow: var(--box-shadow);
                border-radius: var(--border-radius);
                overflow: hidden;
            }}
            
            .stats-table th, .stats-table td {{
                border: 1px solid #ddd;
                padding: 12px 15px;
                text-align: left;
            }}
            
            .stats-table tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            
            .stats-table th {{
                background-color: var(--primary-color);
                color: white;
                text-transform: uppercase;
                font-size: 0.9em;
                letter-spacing: 1px;
            }}
            
            .stats-table tr:hover {{
                background-color: #e6f7ff;
            }}
            
            /* Sección de código */
            .code-section {{
                margin: 40px 0;
                background: var(--dark-bg);
                border-radius: var(--border-radius);
                padding: 20px;
                color: var(--light-text);
            }}
            
            .code-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 1px solid rgba(255,255,255,0.2);
            }}
            
            .code-title {{
                font-family: 'Consolas', monospace;
                font-weight: bold;
                color: #e74c3c;
            }}
            
            /* Animaciones */
            .animation-section {{
                margin: 40px 0;
                text-align: center;
            }}
            
            .animation-container {{
                background: white;
                border-radius: var(--border-radius);
                box-shadow: var(--box-shadow);
                padding: 20px;
                margin: 20px 0;
            }}
            
            .animation-container video, .animation-container img {{
                max-width: 100%;
                border-radius: 5px;
                margin: 10px auto;
                display: block;
            }}
            
            /* Resumen y métricas */
            .metrics-summary {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                margin: 30px 0;
            }}
            
            .metric-card {{
                background: white;
                border-radius: var(--border-radius);
                box-shadow: var(--box-shadow);
                padding: 20px;
                margin: 10px 0;
                width: calc(25% - 20px);
                text-align: center;
                transition: transform 0.3s ease;
            }}
            
            .metric-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            }}
            
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: var(--primary-color);
                margin: 10px 0;
            }}
            
            .metric-label {{
                font-size: 0.9em;
                color: #777;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            /* Apartado de navegación */
            .toc {{
                background: white;
                border-radius: var(--border-radius);
                box-shadow: var(--box-shadow);
                padding: 20px;
                margin: 30px 0;
            }}
            
            .toc ul {{
                list-style-type: none;
                padding-left: 0;
            }}
            
            .toc ul ul {{
                padding-left: 20px;
            }}
            
            .toc li {{
                margin: 8px 0;
            }}
            
            .toc a {{
                color: var(--primary-color);
                text-decoration: none;
                transition: color 0.3s ease;
            }}
            
            .toc a:hover {{
                color: var(--accent-color);
                text-decoration: underline;
            }}
            
            /* Pie de página */
            .footer {{
                margin-top: 50px;
                text-align: center;
                font-size: 0.9em;
                color: #7f8c8d;
                border-top: 1px solid #ddd;
                padding-top: 20px;
            }}
            
            /* Media queries para responsive */
            @media (max-width: 768px) {{
                .metric-card {{
                    width: calc(50% - 20px);
                }}
            }}
            
            @media (max-width: 480px) {{
                .metric-card {{
                    width: 100%;
                }}
            }}
            
            /* CSS para código Python destacado */
            {code_css}
            
            /* Gráficos interactivos */
            .interactive-section {{
                background: white;
                border-radius: var(--border-radius);
                box-shadow: var(--box-shadow);
                padding: 20px;
                margin: 30px 0;
            }}
            
            .tab-container {{
                margin-top: 20px;
            }}
            
            .tab-buttons {{
                display: flex;
                overflow-x: auto;
                border-bottom: 1px solid #ddd;
            }}
            
            .tab-btn {{
                padding: 10px 20px;
                background: none;
                border: none;
                cursor: pointer;
                font-size: 1em;
                color: #555;
                position: relative;
            }}
            
            .tab-btn.active {{
                color: var(--primary-color);
                font-weight: bold;
            }}
            
            .tab-btn.active::after {{
                content: '';
                position: absolute;
                bottom: -1px;
                left: 0;
                width: 100%;
                height: 3px;
                background-color: var(--primary-color);
            }}
            
            .tab-content {{
                padding: 20px 0;
                display: none;
            }}
            
            .tab-content.active {{
                display: block;
            }}
        </style>
    </head>
    <body>
        <h1>Análisis Avanzado de Trayectorias de Patos</h1>
        
        <div class="toc">
            <h3>Contenido</h3>
            <ul>
                <li><a href="#resumen">1. Resumen Ejecutivo</a></li>
                <li><a href="#visualizaciones">2. Visualizaciones de Trayectorias</a>
                    <ul>
                        <li><a href="#trayectorias-2d">2.1 Trayectorias 2D</a></li>
                        <li><a href="#trayectorias-3d">2.2 Trayectorias 3D</a></li>
                        <li><a href="#mapas-calor">2.3 Mapas de Calor y Densidad</a></li>
                    </ul>
                </li>
                <li><a href="#animaciones">3. Animaciones del Movimiento</a></li>
                <li><a href="#estadisticas">4. Análisis Estadístico</a>
                    <ul>
                        <li><a href="#metricas-individuales">4.1 Métricas por Individuo</a></li>
                        <li><a href="#patrones-globales">4.2 Patrones Globales</a></li>
                    </ul>
                </li>
                <li><a href="#codigo">5. Código Fuente</a></li>
                <li><a href="#conclusiones">6. Conclusiones</a></li>
            </ul>
        </div>
        
        <section id="resumen">
            <h2>1. Resumen Ejecutivo</h2>
            
            <div class="metrics-summary">
                <div class="metric-card">
                    <div class="metric-label">Total de Frames</div>
                    <div class="metric-value">{total_frames}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Total de Patos</div>
                    <div class="metric-value">{total_ducks}</div>
                </div>
    """
    
    # Añadir métricas adicionales si hay estadísticas
    if stats_df is not None:
        avg_distance = stats_df['total_distance'].mean()
        avg_speed = stats_df['average_speed'].mean()
        
        html_content += f"""
                <div class="metric-card">
                    <div class="metric-label">Distancia Promedio</div>
                    <div class="metric-value">{avg_distance:.1f}</div>
                    <div>píxeles</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Velocidad Promedio</div>
                    <div class="metric-value">{avg_speed:.2f}</div>
                    <div>píxeles/frame</div>
                </div>
        """
    
    html_content += """
            </div>
            
            <p>Este informe presenta un análisis completo del movimiento de patos capturados mediante el sistema de seguimiento automático Duck-Tracker. Las visualizaciones y análisis estadísticos revelan patrones de desplazamiento, áreas de concentración y diferencias de comportamiento entre individuos.</p>
        </section>
        
        <section id="visualizaciones">
            <h2>2. Visualizaciones de Trayectorias</h2>
    """
    
    # Añadir visualizaciones 2D
    html_content += """
            <section id="trayectorias-2d">
                <h3>2.1 Trayectorias 2D</h3>
                <p>Las siguientes visualizaciones muestran las trayectorias de los patos proyectadas en el plano bidimensional, permitiendo analizar los patrones de movimiento horizontal.</p>
    """
    
    # Añadir visualizaciones que existen
    for filename, caption, description in visualization_files:
        if "2d" in filename.lower() and os.path.exists(os.path.join(visualizations_folder, filename)):
            html_content += f"""
                <div class="visualization-card">
                    <img src="../visualizations/{filename}" alt="{caption}">
                    <div class="card-caption">{caption}</div>
                    <div class="card-description">{description}</div>
                </div>
            """
    
    # Añadir visualizaciones 3D
    html_content += """
            </section>
            
            <section id="trayectorias-3d">
                <h3>2.2 Trayectorias 3D</h3>
                <p>Las visualizaciones tridimensionales incorporan el tiempo como tercera dimensión, mostrando la evolución temporal del movimiento y facilitando la identificación de patrones de comportamiento a lo largo del video.</p>
    """
    
    for filename, caption, description in visualization_files:
        if "3d" in filename.lower() and os.path.exists(os.path.join(visualizations_folder, filename)):
            html_content += f"""
                <div class="visualization-card">
                    <img src="../visualizations/{filename}" alt="{caption}">
                    <div class="card-caption">{caption}</div>
                    <div class="card-description">{description}</div>
                </div>
            """
    
    # Añadir mapas de calor
    html_content += """
            </section>
            
            <section id="mapas-calor">
                <h3>2.3 Mapas de Calor y Densidad</h3>
                <p>Los mapas de calor y densidad revelan las áreas de mayor concentración y actividad de los patos, identificando zonas de interés y patrones de uso del espacio.</p>
    """
    
    for filename, caption, description in visualization_files:
        if ("calor" in filename.lower() or "densidad" in filename.lower()) and os.path.exists(os.path.join(visualizations_folder, filename)):
            html_content += f"""
                <div class="visualization-card">
                    <img src="../visualizations/{filename}" alt="{caption}">
                    <div class="card-caption">{caption}</div>
                    <div class="card-description">{description}</div>
                </div>
            """
    
    html_content += """
            </section>
        </section>
        
        <section id="animaciones">
            <h2>3. Animaciones del Movimiento</h2>
            <p>Las siguientes animaciones muestran la evolución temporal de las trayectorias de los patos, permitiendo apreciar el movimiento dinámico y los patrones de comportamiento a lo largo del tiempo.</p>
    """
    
    # Añadir animaciones
    for anim_file in animation_files:
        if os.path.exists(anim_file):
            file_basename = os.path.basename(anim_file)
            extension = os.path.splitext(file_basename)[1].lower()
            
            if extension == '.mp4':
                html_content += f"""
                <div class="animation-container">
                    <h3>Animación de Trayectorias (MP4)</h3>
                    <video width="800" height="600" controls>
                        <source src="../visualizations/{file_basename}" type="video/mp4">
                        Tu navegador no soporta el tag de video.
                    </video>
                    <p class="card-description">Animación 3D que muestra la evolución de las trayectorias de los patos a lo largo del tiempo. La vista rota para facilitar la apreciación de la estructura tridimensional del movimiento.</p>
                </div>
                """
            elif extension == '.gif':
                html_content += f"""
                <div class="animation-container">
                    <h3>Animación de Trayectorias (GIF)</h3>
                    <img src="../visualizations/{file_basename}" alt="Animación GIF de trayectorias">
                    <p class="card-description">Versión GIF de la animación de trayectorias, optimizada para compatibilidad web y fácil compartición.</p>
                </div>
                """
    
    html_content += """
        </section>
        
        <section id="estadisticas">
            <h2>4. Análisis Estadístico</h2>
            
            <section id="metricas-individuales">
                <h3>4.1 Métricas por Individuo</h3>
                <p>La siguiente tabla muestra las métricas detalladas de movimiento para cada pato detectado, permitiendo comparar el comportamiento individual.</p>
    """
    
    # Añadir tabla de estadísticas si está disponible
    if stats_df is not None:
        # Ordenar por distancia total
        stats_df_sorted = stats_df.sort_values('total_distance', ascending=False)
        
        # Seleccionar columnas relevantes y formatear
        display_columns = ['duck_id', 'color', 'num_frames', 'total_distance', 
                          'average_speed', 'max_speed', 'duration', 'efficiency']
        
        # Verificar qué columnas están disponibles
        available_columns = [col for col in display_columns if col in stats_df_sorted.columns]
        
        # Crear tabla HTML
        html_content += """
                <table class="stats-table">
                    <tr>
        """
        
        # Encabezados de columna
        column_headers = {
            'duck_id': 'ID Pato', 
            'color': 'Color Original', 
            'num_frames': 'Frames', 
            'total_distance': 'Distancia Total (px)', 
            'average_speed': 'Vel. Promedio (px/frame)', 
            'max_speed': 'Vel. Máxima (px/frame)', 
            'duration': 'Duración (frames)',
            'efficiency': 'Eficiencia (%)',
            'main_direction': 'Dir. Principal',
            'dir_percentage': 'Dir. Principal (%)'
        }
        
        for col in available_columns:
            header = column_headers.get(col, col.replace('_', ' ').title())
            html_content += f"<th>{header}</th>"
        
        html_content += """
                    </tr>
        """
        
        # Filas de datos
        for _, row in stats_df_sorted.iterrows():
            html_content += "<tr>"
            for col in available_columns:
                value = row[col]
                # Formatear valores numéricos
                if col in ['total_distance', 'average_speed', 'max_speed']:
                    formatted_value = f"{value:.2f}"
                elif col in ['efficiency', 'dir_percentage']:
                    formatted_value = f"{value*100:.1f}%" if col == 'efficiency' and value <= 1 else f"{value:.1f}%"
                elif col in ['num_frames', 'duration']:
                    formatted_value = f"{int(value)}"
                else:
                    formatted_value = str(value)
                    
                html_content += f"<td>{formatted_value}</td>"
            html_content += "</tr>"
        
        html_content += """
                </table>
        """
    
    # Añadir gráficos estadísticos
    html_content += """
            </section>
            
            <section id="patrones-globales">
                <h3>4.2 Patrones Globales</h3>
                <p>Los siguientes gráficos muestran análisis estadísticos agregados de los patrones de movimiento, revelando tendencias generales y correlaciones entre diferentes métricas.</p>
    """
    
    # Añadir visualizaciones estadísticas
    for filename, caption, description in visualization_files:
        if any(keyword in filename.lower() for keyword in ['velocidad', 'distancia', 'correlacion', 'direcciones']) and os.path.exists(os.path.join(visualizations_folder, filename)):
            html_content += f"""
                <div class="visualization-card">
                    <img src="../visualizations/{filename}" alt="{caption}">
                    <div class="card-caption">{caption}</div>
                    <div class="card-description">{description}</div>
                </div>
            """
    
    html_content += """
            </section>
        </section>
        
        <section id="codigo">
            <h2>5. Código Fuente</h2>
            <p>A continuación se muestra el código fuente utilizado para generar las visualizaciones y análisis de este informe.</p>
    """
    
    # Añadir código fuente con formato destacado
    for filename, code in code_snippets.items():
        highlighted_code, _ = format_python_code(code)
        
        html_content += f"""
            <div class="code-section">
                <div class="code-header">
                    <div class="code-title">{filename}</div>
                </div>
                {highlighted_code}
            </div>
        """
    
    html_content += """
        </section>
        
        <section id="conclusiones">
            <h2>6. Conclusiones</h2>
            <p>El análisis detallado de las trayectorias de los patos revela patrones de comportamiento significativos:</p>
            <ul>
    """
    
    # Generar conclusiones basadas en los datos
    if stats_df is not None:
        # Pato con mayor distancia
        max_dist_duck = stats_df.loc[stats_df['total_distance'].idxmax()]
        html_content += f"""
                <li>El pato <strong>{max_dist_duck['duck_id']}</strong> mostró la mayor actividad, recorriendo una distancia total de <strong>{max_dist_duck['total_distance']:.2f}</strong> píxeles.</li>
        """
        
        # Pato más rápido
        max_speed_duck = stats_df.loc[stats_df['average_speed'].idxmax()]
        html_content += f"""
                <li>El pato <strong>{max_speed_duck['duck_id']}</strong> fue el más rápido, con una velocidad promedio de <strong>{max_speed_duck['average_speed']:.2f}</strong> píxeles/frame.</li>
        """
        
        # Dirección predominante global si está disponible
        if 'main_direction' in stats_df.columns:
            direction_counts = stats_df['main_direction'].value_counts()
            if not direction_counts.empty:
                main_dir = direction_counts.index[0]
                dir_percentage = direction_counts[0] / len(stats_df) * 100
                html_content += f"""
                    <li>La dirección predominante de movimiento fue <strong>{main_dir}</strong>, observada en el <strong>{dir_percentage:.1f}%</strong> de los patos.</li>
            """
    
    html_content += """
                <li>Las visualizaciones de mapas de calor muestran zonas claramente definidas de alta densidad, sugiriendo áreas de interés o recursos dentro del espacio monitoreado.</li>
                <li>La animación tridimensional revela interacciones temporales entre individuos, mostrando posibles comportamientos sociales o respuestas a estímulos externos.</li>
            </ul>
            
            <p>Estos patrones de movimiento proporcionan información valiosa sobre el comportamiento de los patos en entornos controlados, contribuyendo a una mejor comprensión de su ecología y comportamiento social.</p>
        </section>
        
        <section>
            <h2>7. Visualizaciones Interactivas</h2>
            <p>A continuación se presentan visualizaciones interactivas que permiten explorar los datos de diferentes maneras.</p>
            
            <div class="interactive-section">
                <h3>Selección de Visualizaciones</h3>
                
                <div class="tab-container">
                    <div class="tab-buttons">
                        <button class="tab-btn active" onclick="openTab(event, 'tab-trayectorias')">Trayectorias</button>
                        <button class="tab-btn" onclick="openTab(event, 'tab-velocidades')">Velocidades</button>
                        <button class="tab-btn" onclick="openTab(event, 'tab-densidad')">Mapas de Calor</button>
                    </div>
                    
                    <div id="tab-trayectorias" class="tab-content active">
                        <h4>Trayectorias de Patos</h4>
                        <p>Visualización de las trayectorias completas de todos los patos detectados.</p>
                        <img src="../visualizations/trayectorias_2d_colores.png" alt="Trayectorias 2D" style="width:100%; max-width:800px;">
                    </div>
                    
                    <div id="tab-velocidades" class="tab-content">
                        <h4>Análisis de Velocidades</h4>
                        <p>Comparativa de velocidades promedio y máximas por pato.</p>
                        <img src="../visualizations/velocidad_promedio_por_pato.png" alt="Velocidades" style="width:100%; max-width:800px;">
                    </div>
                    
                    <div id="tab-densidad" class="tab-content">
                        <h4>Mapas de Calor</h4>
                        <p>Visualización de áreas de mayor concentración de patos.</p>
                        <img src="../visualizations/mapa_densidad_trayectorias_colores.png" alt="Mapa de Calor" style="width:100%; max-width:800px;">
                    </div>
                </div>
            </div>
        </section>
        
        <div class="footer">
            <p>Informe generado con el sistema Duck-Tracker | Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
            <p>Desarrollado para análisis avanzado de patrones de movimiento en patos</p>
        </div>
        
        <script>
            function openTab(evt, tabName) {{
                // Ocultar todos los contenidos de pestañas
                var tabContents = document.getElementsByClassName("tab-content");
                for (var i = 0; i < tabContents.length; i++) {{
                    tabContents[i].style.display = "none";
                    tabContents[i].className = tabContents[i].className.replace(" active", "");
                }}
                
                // Desactivar todos los botones
                var tabButtons = document.getElementsByClassName("tab-btn");
                for (var i = 0; i < tabButtons.length; i++) {{
                    tabButtons[i].className = tabButtons[i].className.replace(" active", "");
                }}
                
                // Mostrar el contenido de la pestaña actual y activar el botón
                document.getElementById(tabName).style.display = "block";
                document.getElementById(tabName).className += " active";
                evt.currentTarget.className += " active";
            }}
        </script>
    </body>
    </html>
    """
    
    # Guardar el archivo HTML
    output_file = os.path.join(output_folder, 'informe_avanzado_patos.html')
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Informe HTML avanzado generado en: {output_file}")
    
    return output_file

def generate_additional_visualizations(data_file, output_folder):
    """
    Genera visualizaciones adicionales para enriquecer el informe
    
    Args:
        data_file: Ruta al archivo JSON con los datos combinados
        output_folder: Carpeta donde guardar las visualizaciones
    """
    # Crear carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    # Cargar datos para obtener información general
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Extraer datos de frames
    frames_data = data.get('frames', {})
    
    # Preparar estructuras para almacenar trayectorias
    duck_trajectories = {}
    duck_colors = {}
    
    # Procesar cada frame para extraer trayectorias
    for frame_idx, frame_data in sorted(frames_data.items(), key=lambda x: int(x[0])):
        frame_number = int(frame_data.get('frame_number', frame_idx))
        positions = frame_data.get('positions', {})
        
        for duck_id, duck_data in positions.items():
            if duck_id not in duck_trajectories:
                duck_trajectories[duck_id] = []
                duck_colors[duck_id] = duck_data.get('color', 'yellow')
            
            position = duck_data.get('position')
            if position:
                duck_trajectories[duck_id].append({
                    'x': position[0],
                    'y': position[1],
                    'frame': frame_number
                })
    
    # Calcular estadísticas de movimiento
    duck_stats = []
    for duck_id, trajectory in duck_trajectories.items():
        if len(trajectory) > 1:
            # Convertir a arrays numpy para cálculos
            positions = np.array([[point['x'], point['y']] for point in trajectory])
            frames = np.array([point['frame'] for point in trajectory])
            
            # Calcular distancias entre puntos consecutivos
            deltas = positions[1:] - positions[:-1]
            distances = np.sqrt(np.sum(deltas**2, axis=1))
            total_distance = np.sum(distances)
            
            # Calcular velocidades
            frame_deltas = frames[1:] - frames[:-1]
            speeds = distances / np.maximum(frame_deltas, 1)  # Evitar división por cero
            avg_speed = np.mean(speeds)
            max_speed = np.max(speeds)
            
            # Calcular ángulos de movimiento
            angles = np.arctan2(deltas[:, 1], deltas[:, 0]) * 180 / np.pi
            
            # Añadir estadísticas
            duck_stats.append({
                'duck_id': duck_id,
                'color': duck_colors[duck_id],
                'total_distance': total_distance,
                'average_speed': avg_speed,
                'max_speed': max_speed,
                'num_frames': len(trajectory),
                'duration': frames[-1] - frames[0] + 1,
                'angles': angles
            })
    
    # 1. Gráfico de distancia total por pato
    if duck_stats:
        plt.figure(figsize=(12, 8))
        df = pd.DataFrame(duck_stats)
        df = df.sort_values('total_distance', ascending=False)
        
        sns.set_style("whitegrid")
        ax = sns.barplot(x='duck_id', y='total_distance', data=df, palette='viridis')
        plt.title('Distancia Total Recorrida por Pato', fontsize=16)
        plt.xlabel('ID del Pato', fontsize=14)
        plt.ylabel('Distancia Total (píxeles)', fontsize=14)
        plt.xticks(rotation=45)
        
        # Añadir valores en las barras
        for i, v in enumerate(df['total_distance']):
            ax.text(i, v + 5, f"{v:.1f}", ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'distancia_total_por_pato.png'), dpi=300)
        plt.close()
        
        # 2. Gráfico de velocidad promedio por pato
        plt.figure(figsize=(12, 8))
        df = df.sort_values('average_speed', ascending=False)
        
        ax = sns.barplot(x='duck_id', y='average_speed', data=df, palette='plasma')
        plt.title('Velocidad Promedio por Pato', fontsize=16)
        plt.xlabel('ID del Pato', fontsize=14)
        plt.ylabel('Velocidad Promedio (píxeles/frame)', fontsize=14)
        plt.xticks(rotation=45)
        
        # Añadir valores en las barras
        for i, v in enumerate(df['average_speed']):
            ax.text(i, v + 0.05, f"{v:.2f}", ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'velocidad_promedio_por_pato.png'), dpi=300)
        plt.close()
        
        # 3. Gráfico de dispersión velocidad vs distancia
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='total_distance', y='average_speed', 
                       size='num_frames', hue='duck_id',
                       sizes=(100, 500), palette='viridis',
                       data=df)
        
        plt.title('Relación entre Velocidad Promedio y Distancia Total', fontsize=16)
        plt.xlabel('Distancia Total (píxeles)', fontsize=14)
        plt.ylabel('Velocidad Promedio (píxeles/frame)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Añadir etiquetas para cada punto
        for i, row in df.iterrows():
            plt.text(row['total_distance'] + 10, row['average_speed'],
                    row['duck_id'], fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'velocidad_vs_distancia.png'), dpi=300)
        plt.close()
        
        # 4. Gráfico de duración por pato
        plt.figure(figsize=(12, 8))
        df = df.sort_values('duration', ascending=False)
        
        ax = sns.barplot(x='duck_id', y='duration', data=df, palette='magma')
        plt.title('Duración de Aparición en Video por Pato', fontsize=16)
        plt.xlabel('ID del Pato', fontsize=14)
        plt.ylabel('Duración (frames)', fontsize=14)
        plt.xticks(rotation=45)
        
        # Añadir valores en las barras
        for i, v in enumerate(df['duration']):
            ax.text(i, v + 5, f"{int(v)}", ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'duracion_por_pato.png'), dpi=300)
        plt.close()
        
        # 5. Diagrama de rosa de direcciones de movimiento
        plt.figure(figsize=(10, 10), facecolor='white')
        
        # Convertir todos los ángulos a un solo array
        all_angles = []
        for stats in duck_stats:
            all_angles.extend(stats['angles'])
        
        # Crear diagrama de rosa
        ax = plt.subplot(111, projection='polar')
        bins = 16  # Divisiones para las direcciones
        
        # Histograma circular
        heights, edges = np.histogram(all_angles, bins=np.linspace(-180, 180, bins+1))
        width = 2 * np.pi / bins
        bars = ax.bar(np.deg2rad(edges[:-1]), heights, width=width, bottom=0.0)
        
        # Colorear barras según altura
        cm = plt.cm.plasma
        max_height = max(heights)
        for i, bar in enumerate(bars):
            bar.set_facecolor(cm(heights[i]/max_height))
            bar.set_alpha(0.8)
        
        # Configurar gráfico
        ax.set_theta_zero_location('N')  # 0 grados en el Norte
        ax.set_theta_direction(-1)  # Sentido horario
        ax.set_title('Distribución de Direcciones de Movimiento', fontsize=16, pad=20)
        
        # Etiquetas cardinales
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        angles = np.linspace(0, 2*np.pi, len(directions), endpoint=False)
        ax.set_xticks(angles)
        ax.set_xticklabels(directions, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'direcciones_movimiento.png'), dpi=300)
        plt.close()
        
        # 6. Matriz de correlación entre métricas
        plt.figure(figsize=(10, 8))
        
        # Seleccionar columnas numéricas
        numeric_cols = ['total_distance', 'average_speed', 'max_speed', 'num_frames', 'duration']
        corr_matrix = df[numeric_cols].corr()
        
        # Crear mapa de calor
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlación entre Métricas de Movimiento', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'correlacion_metricas.png'), dpi=300)
        plt.close()
        
        # Guardar estadísticas en CSV
        stats_file = os.path.join(output_folder, 'estadisticas_patos.csv')
        df[['duck_id', 'color', 'total_distance', 'average_speed', 'max_speed', 'num_frames', 'duration']].to_csv(stats_file, index=False)
        
        print(f"Visualizaciones adicionales generadas en: {output_folder}")
        
        # Crear resumen global
        resumen_file = os.path.join(output_folder, 'resumen_global.txt')
        with open(resumen_file, 'w') as f:
            f.write(f"RESUMEN ESTADÍSTICO GLOBAL\n")
            f.write(f"=========================\n\n")
            f.write(f"Total de patos analizados: {len(duck_stats)}\n")
            f.write(f"Distancia total promedio: {df['total_distance'].mean():.2f} píxeles\n")
            f.write(f"Velocidad promedio global: {df['average_speed'].mean():.2f} píxeles/frame\n")
            f.write(f"Velocidad máxima registrada: {df['max_speed'].max():.2f} píxeles/frame (Pato {df.loc[df['max_speed'].idxmax(), 'duck_id']})\n")
            f.write(f"Duración promedio en video: {df['duration'].mean():.1f} frames\n\n")
            
            # Pato más activo
            most_active = df.loc[df['total_distance'].idxmax()]
            f.write(f"Pato más activo: {most_active['duck_id']} (Distancia: {most_active['total_distance']:.2f} píxeles)\n")
            
            # Pato más rápido
            fastest = df.loc[df['average_speed'].idxmax()]
            f.write(f"Pato más rápido: {fastest['duck_id']} (Velocidad: {fastest['average_speed']:.2f} píxeles/frame)\n")
            
            # Pato con mayor duración
            longest = df.loc[df['duration'].idxmax()]
            f.write(f"Pato con mayor tiempo en video: {longest['duck_id']} ({int(longest['duration'])} frames)\n")
        
        print(f"Resumen global guardado en: {resumen_file}")

if __name__ == "__main__":
    # Ruta al archivo de datos combinados
    data_file = "/home/alfonso/Duck-Tracker/batch_output/merged_results/merged_tracking_data.json"
    
    # Carpeta con las visualizaciones
    visualizations_folder = "/home/alfonso/Duck-Tracker/batch_output/visualizations"
    
    # Carpeta para guardar el informe
    output_folder = "/home/alfonso/Duck-Tracker/batch_output/informe"
    
    # Lista de archivos de código a incluir
    code_files = [
        "/home/alfonso/Duck-Tracker/visualize_trajectories.py",
        "/home/alfonso/Duck-Tracker/create_animation.py"
    ]
    
    # Generar visualizaciones adicionales
    generate_additional_visualizations(data_file, visualizations_folder)
    
    # Generar informe HTML mejorado
    report_file = create_enhanced_html_report(
        data_file, 
        visualizations_folder, 
        output_folder,
        code_files=code_files
    )
    
    print(f"\n¡Informe avanzado generado con éxito en {report_file}!")
    print("Incluye visualizaciones mejoradas, código fuente y análisis estadísticos detallados.")
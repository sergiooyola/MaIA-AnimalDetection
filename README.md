# MaIA - Animal Detection

Sistema de detección y conteo automático de fauna africana en imágenes aéreas utilizando Deep Learning.

## Objetivo del Proyecto

La conservación de la biodiversidad en África subsahariana enfrenta desafíos críticos debido al incremento de la población humana, conflictos por recursos naturales y la necesidad de compatibilizar la fauna silvestre con actividades productivas. Este proyecto desarrolla un sistema automatizado de monitoreo ecológico que permite:

- **Detectar y contar** múltiples especies de grandes mamíferos africanos en imágenes aéreas de alta resolución.
- **Clasificar** seis especies: Elefante, Búfalo, Kob, Alcelaphinae, Warthog y Waterbuck.
- **Optimizar** el proceso de monitoreo, reemplazando el conteo manual costoso y subjetivo.
- **Fortalecer** la toma de decisiones en conservación y manejo territorial.

### Desafíos Técnicos Abordados

El sistema enfrenta condiciones complejas propias de ecosistemas africanos:
- Animales extremadamente pequeños en imágenes de gran formato (6000×4000 px).
- Alta oclusión y agrupamiento en manadas densas.
- Desbalance taxonómico significativo entre especies.
- Fondos heterogéneos con bajo contraste y vegetación variable.

## Estructura del Repositorio

```
MaIA-AnimalDetection/
│
├── animal-detection/          # Aplicación web de despliegue
│   ├── app.py                # Backend Flask con pipeline de inferencia
│   ├── Dockerfile            # Configuración para containerización
│   ├── requirements.txt      # Dependencias Python
│   ├── models/               # Modelo YOLOv10m entrenado
│   ├── templates/            # Interfaz HTML (index, resultados, historial)
│   ├── static/               # Recursos estáticos y resultados generados
│   └── groundtruth/          # Datos de referencia para evaluación
│
├── notebooks/                 # Experimentación y análisis
│   ├── Yolo1024.ipynb        # Entrenamiento YOLOv10m con tiles 1024×1024
│   ├── InferenciaYolo10m-1024.ipynb  # Inferencia y validación del modelo
│   ├── Tiles1024Yolo.ipynb   # Preprocesamiento y generación de tiles
│   ├── Yolo1024ComparaHerdnet.ipynb  # Comparativa con modelo baseline
│   ├── analisis_datos/       # Análisis exploratorio del dataset
│   └── Pruebas/              # Experimentos con otros modelos y configuraciones
│
├── data/                      # Datasets procesados
│   ├── train_big/            # Conjunto de entrenamiento
│   ├── val_big/              # Conjunto de validación
│   ├── test_big/             # Conjunto de prueba
│   ├── yolo_tiles_1024/      # Tiles generados con formato YOLO
│   └── groundtruth/          # Anotaciones de referencia
│
├── modelos/                   # Pesos del modelo final
│   └── best.pt               # Modelo YOLOv10m optimizado (1024px)
│
└── Informe Final - Despliegue de Soluciones.pdf
```

## Metodología

### 1. Preprocesamiento
- Subdivisión de imágenes en **tiles** (1024×1024 px).
- Solapamiento variable para preservar individuos en bordes.
- Normalización geométrica y filtrado de ruido.
- Eliminación de tiles negativos (sin anotaciones) para el proceso de entrenamiento.

### 2. Modelos Evaluados

Se compararon **arquitecturas avanzadas de deep learning**:

| Modelo | Características Clave | Resultado |
|--------|----------------------|-----------|
| **HerdNet** | Baseline especializado en conteo denso con mapas FIDT | Referencia: F1=0.835, MAE=1.9 |
| **Faster R-CNN** | Detector de dos etapas con RPN multiescala | F1=0.533, MAE=5.8 |
| **P2PNet** | Predicción directa de puntos con matching húngaro | F1=0.119, MAE=4.29 (validación) |
| **YOLOv10m** | Arquitectura eficiente sin NMS, tiles 1024×1024 | **F1=0.741, MAE=3.9** ✓ |
| **YOLO11** | Última generación con mejoras en backbone | F1=0.72, MAE=5.5 |

**Modelo seleccionado:** YOLOv10m con tiles de 1024×1024 px ofreció el mejor balance entre precisión, recall y eficiencia computacional.

### 3. Evaluación
- **Métricas:** Precision, Recall, F1-score, mAP50, mAP50-95, MAE, RMSE, Accuracy de Conteo.
- **Comparativa:** Alineación con métricas reportadas por HerdNet.
- **Análisis por especie:** Desempeño diferenciado según representación en el dataset.

## Sistema de Despliegue

Aplicación web con **Flask + Docker** que permite:
- Cargar imágenes aéreas de gran formato.
- Inferencia automática mediante pipeline de tiling con overlap ajustable.
- Reconstrucción de detecciones en espacio global con eliminación de duplicados.
- Visualización interactiva de resultados con métricas por especie.
- Comparación contra ground truth cuando está disponible.
- Historial de análisis persistente.

### Instalación y Uso

Ver instrucciones detalladas en [`animal-detection/README.md`](animal-detection/README.md)

**Despliegue rápido con Docker:**
```bash
cd animal-detection
docker build -t maia-animal-detection .
docker run -d -p 5000:5000 maia-animal-detection
```

Acceder a `http://3.239.90.11:5000`

## Resultados Principales

El modelo YOLOv10m (1024px) obtuvo:
- **F1-macro:** 0.741
- **MAE:** 3.9
- **RMSE:** 10.6
- **Accuracy de Conteo:** 54.4%

**Desempeño por especie:**
- Especies mayoritarias (Alcelaphinae, Buffalo, Kob): mAP50 entre 0.898-0.920
- Especies minoritarias (Warthog, Waterbuck): mAP50 entre 0.555-0.642
- Elephant: Desempeño moderado afectado por alta oclusión

## Conclusiones y Trabajo Futuro

El sistema demuestra viabilidad para monitoreo automatizado de fauna aérea, aunque con sensibilidad al desbalance taxonómico y condiciones de oclusión extrema.

**Limitaciones identificadas:**
- Dependencia de estrategia de tiling (trade-off resolución vs. redundancia).
- Desempeño comprometido en especies minoritarias.
- Sensibilidad a variaciones de iluminación y fondos complejos.

**Rutas de mejora:**
1. Incorporar módulos de atención multi-escala para objetos diminutos.
2. Técnicas de augmentación específicas para clases minoritarias.
3. Refinamiento de anotaciones con expertos del dominio.
4. Optimización de inferencia distribuida para escalabilidad operacional.

## Referencias

Este proyecto utiliza componentes del repositorio **YOLOv10**, lanzado bajo licencia **GPL-3.0**.

- [Repositorio YOLOv10 original](https://github.com/THU-MIG/yolov10)
- [Dataset: African Mammal Aerial Imagery - ULiège](https://dataverse.uliege.be/dataset.xhtml?persistentId=doi:10.58119/ULG/MIRUU5)

## Autores

Braulio Martínez, Hada Licette Sandoval, Julio Pachón, Sergio Oyola

---

**Proyecto desarrollado para:** Despliegue de Soluciones de la Maestría en Inteligencia Artificial - Universidad de los Andes (Noviembre 2025)

# README — Proyecto YOLOv10M 1024×1024 (MaIA Animal Detection)

Este repositorio contiene el proceso completo de analisis de data, preprocesamiento, entrenamiento e inferencia para el modelo YOLOv10-M, aplicado al proyecto de detección y conteo de manadas en África utilizando tiles de 1024×1024 píxeles. Adicionalmente a las pruebas con otros modelos.

## Estructura del repositorio en este directorio

```
notebooks/Tiles1024Yolo.ipynb
notebooks/Yolo1024.ipynb
notebooks/InferenciaYolo10m-1024.ipynb
datos/yolo_tiles_1024/
notebooks/tiles_1024_jpg_q100/
notebooks/Pruebas/
```

## 1. Preprocesamiento de imágenes
Archivo: `Tiles1024Yolo.ipynb`

Este notebook realiza las siguientes tareas:

- Carga del dataset original de imágenes grandes.
- Generación de tiles 1024×1024 mediante ventanas deslizantes.
- Ajuste y validación de bordes.
- Transformación de anotaciones al formato YOLO.
- Organización en carpetas de entrenamiento y validación.

Los tiles generados se encuentran en:

```
datos/yolo_tiles_1024/
```

## 2. Entrenamiento del modelo YOLOv10-M
Archivo: `Yolo1024.ipynb`

Este notebook incluye:

- Configuración del modelo YOLOv10-M.
- Entrenamiento utilizando imágenes de 1024×1024.
- Selección y ajuste de hiperparámetros.
- Obtención de métricas de desempeño, incluyendo:
  - mAP50
  - mAP50–95
  - Precisión
  - Recall
- Generación y almacenamiento de resultados y gráficas.

Las métricas y resultados del entrenamiento del modelo escogido se encuentran en:

```
tiles_1024_jpg_q100/
```

## 3. Inferencia y validación
Archivo: `InferenciaYolo10m-1024.ipynb`

Incluye:

- Analisis de datos
- Inferencias sobre imágenes reales del dataset.
- Comparación entre predicciones y anotaciones reales.
- Evaluación del modelo por especie.
- Cálculo de exactitud general.
- Visualización de detecciones y análisis de errores.
- Entrenamiento de pruebas con otros modelos.
- Métricas para comparar con Herdnet.

## Descripción de carpetas

### yolo_tiles/
Contiene todas las imágenes 1024×1024 utilizadas para entrenamiento y validación, junto con sus etiquetas en formato YOLO.

### tiles_1024_jpg_q100/
Incluye las métricas del entrenamiento, gráficas comparativas y resultados por clase.

## Resumen general

Este repositorio documenta el pipeline completo:

1. Preprocesamiento: `Tiles1024Yolo.ipynb`
2. Entrenamiento del modelo YOLOv10-M: `Yolo1024.ipynb`
3. Inferencia y validación: `InferenciaYolo10m-1024.ipynb`

Las imágenes y resultados generados se encuentran organizados en las carpetas `yolo_tiles/` y `tiles_1024_jpg_q100/`.

Ademas de incluir:
1. Analisis de datos en `AnalisisDataJson.ipynb`

## Generacion de metricas comparativas con HerdNet
Para obtener las metricas alineadas con HerdNet, se realizaron las inferencias sobre el dataset de test y con el modelo entrenado Yolov10M tiles: 1024. Se detalla en Yolo1024ComparaHerdnet.ipynb

## Entrenamientos con otros modelos no seleccionados
En el directorio "Pruebas" se encuentran los ipynb de entrenamientos de prueba realizados con otras configuraciones y modelos.

## Licencia y atribuciones
Este proyecto utiliza componentes del repositorio **YOLOv10**, lanzado bajo licencia **GPL-3.0** por sus autores originales.
Repositorio original:  
https://github.com/THU-MIG/yolov10
La redistribución o uso de partes de dicho código mantiene las obligaciones establecidas en la licencia GPL-3.0.


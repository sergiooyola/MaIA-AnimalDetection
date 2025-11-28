# MaIA - Animal Detection (YOLOv8 + Flask + Docker)

Sistema de detección y conteo de fauna silvestre basado en imágenes aéreas.  
El proyecto permite:

- Subir una imagen
- Dividirla en tiles (opcional)
- Detectar animales con YOLOv8
- Reconocer 6 especies
- Comparar contra Ground Truth (GT)
- Calcular métricas (Accuracy, MAE, RMSE, F1 por clase)
- Guardar historial de análisis
- Generar visualizaciones y resultados persistentes

Este repositorio contiene el código listo para despliegue en **Docker** y **AWS EC2**.

---

## 📁 Estructura del Proyecto

animal-detection/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
│
├── models/
│ └── best.pt
│
├── groundtruth/
│ └── test.csv
│
├── templates/
│ ├── index.html
│ ├── resultado.html
│ └── historial.html
│
├── static/
│ ├── resultados/
│ ├── tiles/


## ⚙️ Instalación y Ejecución Local

### 1. Crear ambiente (opcional)

```bash
#python -m venv venv
#source venv/bin/activate

pip install -r requirements.txt

python app.py

http://localhost:5000

docker build -t animal-detection .

docker run -d -p 5000:5000 animal-detection

http://localhost:5000

git clone https://github.com/sergiooyola/MaIA-AnimalDetection.git
cd MaIA-AnimalDetection/animal-detection

Despliegue en AWS EC2

Clonar el repo en la instancia:

git clone https://github.com/sergiooyola/MaIA-AnimalDetection.git
cd MaIA-AnimalDetection/animal-detection

docker build -t animal-detection .

docker run -d -p 5000:5000 animal-detection


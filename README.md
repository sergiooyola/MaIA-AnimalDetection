1. Convertir las anotaciones de csv a json: EDA_Validacion_Export_COCO_Dataset.ipynb
2. Pasar las imagenes a tiles: Genera_tiles.ipynb
3. Entrenar, Probar resultados y estadisticas: Train_Faster_R_CNN_sobre_tiles.ipynb

<strong>Detalle EDA_Validacion_Export_COCO_Dataset.ipynb</strong>
   1) Instalación y carga de librerías
Ejecuta pip install para tener pandas, numpy, matplotlib, seaborn, opencv-python, pycocotools, tqdm.
Importa los paquetes y configura seaborn con estilo “whitegrid”.
Mensaje de control: ✅ Librerías cargadas.

2) Parámetros de entrada (rutas)
Define el tipo de dato (DATA_TYPE = 'csv') y rutas a:
CSV de train, val, test (con columnas: Image, x1, y1, x2, y2, Label).
Carpetas de imágenes (train/val/test).
Carpeta OUTPUT_DIR donde guardará los JSON convertidos a COCO.
Aquí decides desde dónde leer y dónde escribir.

3) Utilidades: carga y conversión a COCO
load_csv(path)
Lee el CSV con pandas.
Limpia nombres de columnas (quita espacios).
Muestra cuántos registros trae: 📄 CSV con N registros.
csv_to_coco(csv_path, images_dir, output_json)
Convierte tu CSV a un JSON en formato COCO:

Construye:
categories: a partir de los Label únicos en el CSV (los mapea a ids 1..K).
images: una entrada por imagen (con id incremental, file_name, width, height).
La función, por la estructura típica, infiere ancho/alto abriendo la imagen o los puede tomar si el CSV lo ofrece (según la versión de tu cuaderno; en el tuyo construye la lista basándose en los registros y asocia cada nombre de imagen con un image_id).
annotations: una entrada por bounding box (con:
image_id, category_id,
bbox en COCO [x, y, w, h] (lo calcula a partir de tus columnas x1,y1,x2,y2),
area = w*h,
iscrowd = 0).
Guarda el JSON final con:
{
  "info": {...},
  "licenses": [],
  "images": [...],
  "annotations": [...],
  "categories": [...]
}
Mensaje de éxito: ✅ Guardado <ruta_del_json>.
Resultado: tendrás, por ejemplo, train_converted.json (y más adelante también los de val y test).

4) EDA de cajas (análisis exploratorio)
analyze_boxes(df)
Calcula anchos y altos a partir de x2-x1 y y2-y1.
Muestra un describe() con count/mean/std/min/percentiles/max para width y height.
Grafica histogramas de distribución de anchos y altos.
suggest_anchors(stats)
Toma la media de width y height y sugiere una rejilla típica de anchor_sizes: [8,16,32,64,128].
Imprime el tamaño promedio y la lista sugerida.
Con esto te haces una idea del tamaño típico de objeto y si tus anchors default son razonables.

5) EDA rápida sobre train y conversión a COCO (train)
df_train = load_csv(TRAIN_PATH)
value_counts() de Label y gráfica de barras con la distribución de clases.
stats = analyze_boxes(df_train) y suggest_anchors(stats) para ver tamaños.
Convierte el CSV de train a COCO:
csv_to_coco(TRAIN_PATH, TRAIN_IMG_DIR, os.path.join(OUTPUT_DIR,'train_converted.json'))

6) Validación de estructura COCO (train)
Carga el JSON convertido con pycocotools:
coco = COCO(os.path.join(OUTPUT_DIR, 'train_converted.json'))

Imprime:
📷 Imágenes: <n_imgs>
📦 Anotaciones: <n_anns>
🐾 Categorías: <n_cats> → [primeras categorías]
Esto verifica que el JSON esté coherente y que COCO lo pueda indexar.

7) Visualización de una imagen con cajas (colores por clase)
Define una paleta de colores por clase (1..6) en BGR para OpenCV:
1: negro
2: rojo
3: azul
4: amarillo
5: cian (azul claro)
6: verde
show_random_image(coco, TRAIN_IMG_DIR):
Toma una imagen aleatoria del COCO.
Recupera sus anotaciones (bboxes y categorías).
Dibuja rectángulos semi-transparentes (con alpha=0.5) del color de la clase (sin texto, solo caja).
Muestra título "<file_name> | <num_boxes> boxes" y la figura.
Te sirve para validar visualmente que las cajas y clases coinciden con lo esperado.

8) Conversión de val y test a COCO
Finalmente convierte VAL y TEST:
csv_to_coco(VAL_PATH, VAL_IMG_DIR, os.path.join(OUTPUT_DIR,'val_converted.json'))
csv_to_coco(TEST_PATH, TEST_IMG_DIR, os.path.join(OUTPUT_DIR,'test_converted.json'))

<strong>Detalle Genera_tiles.ipynb</strong>

🧱 Propósito general del notebook

Dividir las imágenes originales del dataset (de gran tamaño, ~6000×4000 px) en fragmentos más pequeños —tiles— de 1024×1024 px con solape (overlap) de 128 px.
También genera nuevas anotaciones COCO para esos tiles, asegurando que cada caja (bounding box) se recorte correctamente y conserve su etiqueta de clase.

🔹 1. Importación de librerías y configuración
import os, json, math, cv2
from tqdm import tqdm
from pathlib import Path


os, pathlib → manejo de rutas.
json → lectura/escritura de anotaciones COCO.
cv2 → manipulación de imágenes (OpenCV).
tqdm → barra de progreso durante el procesamiento.

🔹 2. Rutas y parámetros principales
SRC_IMG = {
  "train": ".../train",
  "val":   ".../val",
  "test":  ".../test",
}
SRC_JSON = {
  "train": ".../train_converted.json",
  "val":   ".../val_converted.json",
  "test":  ".../test_converted.json",
}
OUT_ROOT = ".../tiles"
TILE = 1024        # tamaño de cada tile
OVERLAP = 128      # solape entre tiles
MIN_VIS_FRAC = 0.3 # fracción mínima visible para conservar una bbox


📌 Qué significa cada uno:
Las rutas apuntan al dataset original (imágenes y JSONs COCO ya convertidos desde CSV).
TILE = 1024 → cada fragmento tendrá 1024×1024 px.
OVERLAP = 128 → se solapan parcialmente para no perder objetos en los bordes.
MIN_VIS_FRAC = 0.3 → una caja que quede cortada debe conservar al menos un 30 % de su área original para incluirse.

🔹 3. Funciones auxiliares
load_json(p)
Carga un archivo JSON (anotaciones COCO).
coco_index(coco)
Convierte la estructura COCO a un diccionario indexado por image_id:

{
  image_id: { "file_name":..., "height":..., "width":..., "anns":[...] }
}
para acceder fácilmente a las anotaciones de cada imagen.


🔹 4. Funciones geométricas
intersect_box(box, win)
Devuelve la intersección entre la caja box y la ventana (tile) win.
Se usa para recortar las cajas que caen parcialmente dentro de un tile.
area(box)
Calcula el área de una caja [x1, y1, x2, y2].
clamp_to_tile(box, tx, ty)
Ajusta coordenadas al sistema local del tile (restando el offset del tile dentro de la imagen).
to_xyxy(b) y to_xywh(b)
Convierte entre formatos de coordenadas:
[x,y,w,h] → [x1,y1,x2,y2]
y viceversa.

🔹 5. Función principal: tile_split(split)
Esta es la parte más importante.
Recorre cada imagen del conjunto split (train, val o test) y crea sus tiles con sus anotaciones.

a) Carga del JSON y prepara salida
coco = load_json(SRC_JSON[split])
idx = coco_index(coco)
out_imgs, out_anns = [], []
out_dir = f"{OUT_ROOT}/{split}_{TILE}_ov{OVERLAP}"
Path(out_dir).mkdir(parents=True, exist_ok=True)

b) Itera sobre cada imagen
Para cada imagen:
Abre el archivo con cv2.imread.
Obtiene h, w.
Recorre la imagen en pasos (step = TILE - OVERLAP), generando ventanas deslizantes.

c) Generación de tiles
for ty in range(0, h, step_y):
    for tx in range(0, w, step_x):
        x2, y2 = min(tx + TILE, w), min(ty + TILE, h)


Crea el recorte tile_img = img[ty:y2, tx:x2].
Lo guarda como archivo JPG en la carpeta de salida.


d) Ajuste de las cajas
Para cada anotación (bounding box) de la imagen original:
Convierte [x, y, w, h] → [x1, y1, x2, y2].
Calcula la intersección con el tile.
Si el área visible (vis_frac) ≥ MIN_VIS_FRAC, se conserva.
Ajusta coordenadas al tile con clamp_to_tile.
Guarda la nueva caja en formato COCO ([x, y, w, h]) con su category_id.
📌 Cada bbox genera una nueva anotación con un id único y el nuevo image_id del tile.


e) Guarda metadatos de cada tile
Si el tile tiene al menos una bbox válida:
out_imgs.append({ "id": new_img_id, "file_name": tile_name, "height": ..., "width": ... })
out_anns.extend(tile_anns_this)
Y suma new_img_id += 1.


🔹 6. Generación del nuevo JSON COCO
Al final de cada split:
out = {
  "info": {"description": f"{split} tiles", "tile": TILE, "overlap": OVERLAP},
  "licenses": [],
  "images": out_imgs,
  "annotations": out_anns,
  "categories": [{"id": i, "name": str(i)} for i in [1,2,3,4,5,6]]
}
with open(f"{OUT_ROOT}/annotations/{split}_tiles.json","w") as f:
    json.dump(out, f, indent=2)

✅ Guarda el archivo train_tiles.json, val_tiles.json, test_tiles.json.


🔹 7. Bucle final
for sp in ["train","val","test"]:
    tile_split(sp)
Ejecuta el proceso completo para los tres conjuntos.
📦 Resultado final
Carpeta /tiles/ con subcarpetas:
tiles/
  ├─ train_1024_ov128/
  ├─ val_1024_ov128/
  ├─ test_1024_ov128/
  └─ annotations/
      ├─ train_tiles.json
      ├─ val_tiles.json
      └─ test_tiles.json
Miles de subimágenes (.jpg) con tamaño ~1024×1024 px.
JSONs COCO actualizados con las anotaciones recortadas y adaptadas.

<strong>Detalle Train_Faster_R_CNN_sobre_tiles.ipynb</strong>

🧩 Propósito general

Entrenar un modelo Faster R-CNN con ResNet-50 + FPNv2 sobre los tiles generados (subimágenes 1024×1024) y sus anotaciones COCO.
El objetivo: detectar animales en cada tile, con entrenamiento desde pesos preentrenados en COCO.

🔹 1. Preparación del entorno

El notebook empieza instalando y configurando las dependencias:

!pip install albumentations==1.4.8 pycocotools==2.0.7 tqdm opencv-python==4.9.0.80 --quiet
!pip install --upgrade --no-cache-dir torch torchvision torchaudio --quiet


Albumentations → para futuras augmentaciones (no se usa directamente aquí, pero está lista).

pycocotools → para manejar anotaciones COCO.

torch / torchvision → red neuronal y modelo de detección.

OpenCV → lectura y visualización de imágenes.

tqdm → progreso en bucles largos.

También se monta Google Drive si es Colab:

from google.colab import drive
drive.mount('/content/drive')


Y se valida que el entorno detecte CUDA (GPU):

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("🚀 Dispositivo:", DEVICE)

🔹 2. Rutas de dataset y modelo

Define las rutas hacia las imágenes tileadas y los JSON COCO:

TRAIN_IMG_DIR = "/content/drive/.../tiles/train_1024_ov128"
VAL_IMG_DIR   = "/content/drive/.../tiles/val_1024_ov128"
TEST_IMG_DIR  = "/content/drive/.../tiles/test_1024_ov128"

TRAIN_JSON = "/content/drive/.../tiles/annotations/train_tiles.json"
VAL_JSON   = "/content/drive/.../tiles/annotations/val_tiles.json"
TEST_JSON  = "/content/drive/.../tiles/annotations/test_tiles.json"


Crea también la carpeta donde se guardarán los modelos:

os.makedirs("/content/drive/.../models", exist_ok=True)

🔹 3. Definición del Dataset personalizado
class CocoTiles(CocoDetection)

Hereda de torchvision.datasets.CocoDetection.

Esta clase adapta el dataset COCO para el formato que espera Faster R-CNN:

def __getitem__(self, idx):
    img, target = super().__getitem__(idx)
    img = F.to_tensor(img)

    boxes, labels = [], []
    for ann in target:
        boxes.append(ann["bbox"])
        labels.append(ann["category_id"])

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    boxes[:, 2:] += boxes[:, :2]  # pasa de [x,y,w,h] a [x1,y1,x2,y2]

    target = {
        "boxes": boxes,
        "labels": torch.as_tensor(labels, dtype=torch.int64),
        "image_id": torch.tensor([idx])
    }
    return img, target


✅ Qué hace:

Lee una imagen y sus anotaciones desde COCO.

Convierte a tensores PyTorch.

Ajusta el formato de cajas.

Devuelve el par (imagen_tensor, diccionario_target).

🔹 4. DataLoaders
train_ds = CocoTiles(TRAIN_IMG_DIR, TRAIN_JSON)
val_ds   = CocoTiles(VAL_IMG_DIR, VAL_JSON)


Se cargan con DataLoader para iterar durante el entrenamiento:

train_dl = DataLoader(train_ds, batch_size=2, shuffle=True,
                      num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
val_dl   = DataLoader(val_ds, batch_size=2, shuffle=False,
                      num_workers=2, collate_fn=lambda x: tuple(zip(*x)))


El collate_fn es esencial porque cada imagen tiene distinto número de cajas; agrupa correctamente los lotes como listas.

🔹 5. Construcción del modelo Faster R-CNN
model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")


Usa ResNet-50 como backbone y Feature Pyramid Network (FPN) como extractor multiescala.

Preentrenado en COCO, por lo tanto ya entiende características visuales generales.

Luego se reemplaza la cabeza de clasificación:

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 7)


👉 7 = 6 clases de animales + 1 fondo (background).

Finalmente:

model.to(DEVICE)

🔹 6. Optimizador y scheduler
opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)


AdamW → buen optimizador con regularización incorporada.

CosineAnnealingLR → ajusta la tasa de aprendizaje suavemente entre épocas.

🔹 7. Funciones de entrenamiento y evaluación
🧠 train_one_epoch(epoch)

Ciclo principal de entrenamiento:

def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    for imgs, tgts in tqdm(train_dl, desc=f"Epoch {epoch:02d}"):
        imgs = [im.to(DEVICE) for im in imgs]
        tgts = [{k:v.to(DEVICE) for k,v in t.items()} for t in tgts]
        loss_dict = model(imgs, tgts)
        loss = sum(loss_dict.values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(train_dl)


✅ Calcula la pérdida de detección (clasificación + regresión) y actualiza pesos.

🔎 evaluate_val()

Valida el modelo en el conjunto de validación con un criterio simple:
compara el número de cajas detectadas vs el número real (Ground Truth).

@torch.no_grad()
def evaluate_val():
    model.eval()
    mae = []
    for imgs, tgts in val_dl:
        imgs = [im.to(DEVICE) for im in imgs]
        outs = model(imgs)
        for o, t in zip(outs, tgts):
            pred_n = (o["scores"] > 0.5).sum().item()
            true_n = len(t["boxes"])
            mae.append(abs(pred_n - true_n))
    return np.mean(mae)


Devuelve el MAE (Mean Absolute Error) del conteo de detecciones.

🔹 8. Entrenamiento principal
EPOCHS = 10
for ep in range(1, EPOCHS + 1):
    loss = train_one_epoch(ep)
    mae = evaluate_val()
    sched.step()
    print(f"[Epoch {ep:02d}] loss={loss:.4f} | MAE={mae:.3f}")


Entrena durante 10 épocas.

Muestra la pérdida media y el error de conteo (MAE).

Durante el entrenamiento se guardan checkpoints cada 2 épocas:

torch.save(model.state_dict(), f"/models/checkpoint_epoch_{ep:02d}.pth")


Y al final:

torch.save(model.state_dict(), "/models/fasterrcnn_animals_tiles_final.pth")

🔹 9. Fine-tuning (entrenamiento adicional)

Después del modelo base, se ejecuta un nuevo bloque de entrenamiento cargando los pesos previos:

model.load_state_dict(torch.load("/models/fasterrcnn_animals_tiles_final.pth"))


Y se entrena 5 épocas más con un learning rate menor, manteniendo el mismo esquema.
Resultado guardado como:

"/models/fasterrcnn_animals_tiles_finetuned.pth"

🔹 10. Evaluación posterior (COCOEval)

El notebook concluye con evaluación completa en el conjunto de validación:

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


Genera un JSON con las detecciones del modelo.

Calcula métricas COCO:

AP@[0.5:0.95], AP@0.5, AP@0.75

AR@[1,10,100]

AP/AR por tamaño (small, medium, large)

Compara modelo base vs fine-tuned.

🔹 11. Resultados obtenidos (según tus logs)
Modelo	AP@[.5:.95]	AP@.5	MAE
Base	0.319	0.557	0.82
Fine-tuned	0.312	0.545	0.87

👉 El modelo mantuvo desempeño similar, con ligera estabilización tras el fine-tuning (menor pérdida, pero sin incremento grande en AP global).

📊 En resumen
Etapa	Descripción
1	Carga dependencias y GPU
2	Define rutas del dataset tileado
3	Implementa clase COCO adaptada
4	Crea DataLoader para entrenamiento y validación
5	Carga Faster R-CNN preentrenado
6	Define optimizador y scheduler
7	Entrena modelo durante 10 épocas
8	Evalúa con MAE (conteo)
9	Fine-tuning (5 épocas más)
10	Evalúa con métricas COCO (AP, AR)
11	Guarda checkpoints y modelo final

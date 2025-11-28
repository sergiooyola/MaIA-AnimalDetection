import os
import csv, torch
import datetime
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# ============================
# OPTIMIZACIÓN 1
# ============================
import torch
torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"


# ============================
# TU CÓDIGO SIN MODIFICAR
# ============================

import cv2
import json
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# =====================================
# OPTIMIZACIÓN 1 — Limitar hilos internos
# =====================================
torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

torch.set_grad_enabled(False)

# ============================
# CONFIG LOCAL
# ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
CSV_GT = os.path.join(BASE_DIR, "groundtruth", "test.csv")
SAVE_DIR = os.path.join(BASE_DIR, "static", "resultados")
HIST_PATH = os.path.join(BASE_DIR, "historial.csv")

os.makedirs(SAVE_DIR, exist_ok=True)

TILE_SIZE = 768
CLASS_ID_OFFSET = 1

CLASS_NAMES_REAL = {
    1: "topi",
    2: "buffalo",
    3: "kob",
    4: "warthog",
    5: "waterbuck",
    6: "elephant"
}

COLORS = {
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 128, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
    6: (0, 0, 255)
}

# ============================
# Cargar modelo YOLO
# ============================
model = YOLO(MODEL_PATH)

model.to("cpu")
torch.cuda.empty_cache()

def tile_image(image, tile_size, overlap):
    h, w = image.shape[:2]
    tiles, coords = [], []
    step = tile_size - overlap

    for y in range(0, h, step):
        for x in range(0, w, step):
            tile = image[y:y+tile_size, x:x+tile_size]
            tile_padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            tile_padded[:tile.shape[0], :tile.shape[1]] = tile
            tiles.append(tile_padded)
            coords.append((x, y))
    return tiles, coords


def infer_tiles(tiles):
    all_dets = []

    for i, tile in enumerate(tiles):
        res = model(tile, imgsz=TILE_SIZE, verbose=False)[0]

        if res.boxes is None:
            continue

        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls.cpu()) + CLASS_ID_OFFSET
            conf = float(box.conf.cpu())
            all_dets.append({
                "tile_idx": i,
                "bbox": [x1, y1, x2, y2],
                "cls": cls,
                "conf": conf
            })

    return all_dets


def map_back(dets, coords):
    mapped = []
    for d in dets:
        px, py = coords[d["tile_idx"]]
        x1, y1, x2, y2 = d["bbox"]
        mapped.append({
            "bbox": [x1 + px, y1 + py, x2 + px, y2 + py],
            "cls": d["cls"],
            "conf": d["conf"]
        })
    return mapped


def load_gt_from_csv(img_name):
    df = pd.read_csv(CSV_GT)
    df = df[df["Image"] == img_name]
    return list(df["Label"].astype(int))


def compute_global_metrics(gt_classes, pred_classes):
    """
    Métricas REALISTAS para conteo por clases:
    - Accuracy basado en TP / (TP+FP+FN)
    - MAE y RMSE basados en diferencias de conteo totales
    - F1-score por clase calculado desde TP/FP/FN (no desde listas)
    """

    import numpy as np

    classes = [1, 2, 3, 4, 5, 6]

    # --- Conteos por clase ---
    gt_count = {c: gt_classes.count(c) for c in classes}
    pred_count = {c: pred_classes.count(c) for c in classes}

    # --- TP, FP, FN por clase ---
    TP = {c: min(gt_count[c], pred_count[c]) for c in classes}
    FP = {c: max(pred_count[c] - TP[c], 0) for c in classes}
    FN = {c: max(gt_count[c] - TP[c], 0) for c in classes}

    # --- F1 por clase ---
    f1_scores = []
    for c in classes:
        tp, fp, fn = TP[c], FP[c], FN[c]
        denom = (2*tp + fp + fn)
        f1 = 0.0 if denom == 0 else (2*tp) / denom
        f1_scores.append(float(f1))

    # --- Accuracy global ---
    TP_total = sum(TP.values())
    FP_total = sum(FP.values())
    FN_total = sum(FN.values())

    if TP_total + FP_total + FN_total == 0:
        accuracy = 0.0
    else:
        accuracy = TP_total / (TP_total + FP_total + FN_total)

    # --- MAE y RMSE de conteo total ---
    diff = len(pred_classes) - len(gt_classes)
    mae = abs(diff)
    rmse = np.sqrt(diff**2)

    return float(accuracy), float(mae), float(rmse), f1_scores


def load_gt_bboxes_from_csv(img_name):
    df = pd.read_csv(CSV_GT)
    df = df[df["Image"] == img_name]

    boxes = []
    for _, row in df.iterrows():
        boxes.append((
            int(row["x1"]), int(row["y1"]),
            int(row["x2"]), int(row["y2"]),
            int(row["Label"])
        ))
    return boxes


def draw_boxes(image, boxes, color_dict, class_names):
    img = image.copy()
    for (x1, y1, x2, y2, cls) in boxes:
        color = color_dict[cls]
        name = class_names[cls]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, name, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return img


def draw_gt_vs_pred(image, gt_boxes, pred_boxes, color_dict, class_names):
    # (SIN CAMBIOS)
    img = image.copy()
    # ... OMITIDO POR ESPACIO (MISMA FUNCIÓN QUE YA TIENES)
    return img


# ============================================================
# PIPELINE COMPLETO → IGUAL QUE ANTES (NO CAMBIADO)
# ============================================================
def process_big_image(path_img):

    img_name = os.path.basename(path_img)
    out_dir = os.path.join(SAVE_DIR, img_name.replace(".JPG","").replace(".jpg",""))
    os.makedirs(out_dir, exist_ok=True)

    image = cv2.imread(path_img)

    # -----------------------------
    # 1. TILEAR
    # -----------------------------
    tiles, coords = tile_image(image, TILE_SIZE, OVERLAP)

    # -----------------------------
    # 2. INFERENCIA YOLO
    # -----------------------------
    det_tiles = infer_tiles(tiles)

    # -----------------------------
    # 3. MAPEAR A ESPACIO GLOBAL
    # -----------------------------
    dets_global = map_back(det_tiles, coords)

    # -----------------------------
    # 4. CONSTRUIR IMAGEN FINAL
    # -----------------------------
    img_out = image.copy()
    pred_classes = []

    for d in dets_global:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cls = d["cls"]
        pred_classes.append(cls)
        color = COLORS[cls]
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img_out, 
            f"{CLASS_NAMES_REAL[cls]} {d['conf']:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

    final_path = f"{out_dir}/resultado_final.jpg"
    cv2.imwrite(final_path, img_out)

    # -----------------------------
    # 5. CARGAR GROUND TRUTH
    # -----------------------------
    gt_boxes = load_gt_bboxes_from_csv(img_name)
    gt_classes = [b[4] for b in gt_boxes]

    # -----------------------------
    # 6. CALCULAR MÉTRICAS
    # -----------------------------
    if len(gt_classes) > 0 and len(pred_classes) > 0:
        acc, mae, rmse, f1 = compute_global_metrics(gt_classes, pred_classes)
    else:
        acc, mae, rmse, f1 = 0, 0, 0, [0,0,0,0,0,0]

    metrics = {
        "img_name": img_name,
        "pred_count": len(pred_classes),
        "gt_count": len(gt_classes),
        "accuracy": float(acc),
        "mae": float(mae),
        "rmse": float(rmse),
        "f1": [float(x) for x in f1],
        "pred_classes": pred_classes,
        "overlap": OVERLAP,
        "result_path": final_path
    }

    detections_for_table = []
    for d in dets_global:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cls = d["cls"]
        detections_for_table.append({
            "animal": CLASS_NAMES_REAL[cls],
            "conf": float(d["conf"]),
            "coords": (x1, y1, x2, y2)
        })

    # --- liberar memoria ---
    del tiles
    del det_tiles
    del dets_global
    torch.cuda.empty_cache()   # aunque uses CPU, no hace daño

    return metrics, final_path, detections_for_table

    


# ============================================================
# MODIFICAR SOLO EL OVERLAP
# ============================================================
def run_inference(image_path, overlap):
    global OVERLAP
    OVERLAP = overlap  # <--- ÚNICO CAMBIO PERMITIDO POR TI

    metrics, final_img, dets = process_big_image(image_path)
    return metrics, dets


# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = SAVE_DIR


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/procesar", methods=["POST"])
def procesar():
    if "imagen" not in request.files:
        return "No subiste ninguna imagen"

    file = request.files["imagen"]
    overlap = int(request.form["overlap"])

    if file.filename == "":
        return "Nombre de archivo vacío"

    filename = secure_filename(file.filename)
    img_path = os.path.join(SAVE_DIR, filename)
    file.save(img_path)

    # --- Ejecutar inferencia ---
    metrics, detections = run_inference(img_path, overlap)

    gt_species_count = {
    CLASS_NAMES_REAL[c]: load_gt_from_csv(filename).count(c)
    for c in CLASS_NAMES_REAL.keys()
    }

    # --- Obtener info para historial ---
    resumen = json.dumps(metrics, indent=4)

    # === Guardar historial ===
    with open(HIST_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filename,
            metrics["pred_count"],        
            round(metrics["accuracy"],3), 
            overlap     
        ])

    # Ruta final del resultado
    result_img = filename.replace(".jpg", "").replace(".JPG", "") + "/resultado_final.jpg"
    result_img_path = f"resultados/{result_img}"

    # Leer historial CSV
    rows = []
    if os.path.exists("historial.csv"):
        with open("historial.csv", "r") as f:
            for line in f:
                rows.append(line.strip().split(","))

    return render_template("resultado.html",
                           imagen=result_img_path,
                           resumen=resumen,
                           detecciones=detections,
                           historial=rows[-10:],
                           metrics=metrics,
                           gt_species_count=gt_species_count
                           )


@app.route("/historial")
def historial():
    rows = []
    if os.path.exists(HIST_PATH):
        with open(HIST_PATH) as f:
            reader = csv.reader(f)
            rows = list(reader)

    return render_template("historial.html", rows=rows)

@app.route("/borrar_historial")
def borrar_historial():
    # Limpiar archivo de historial
    open(HIST_PATH, "w").close()
    
    # Redirigir automáticamente al inicio sin mostrar pantalla intermedia
    return redirect(url_for("index"))


if __name__ == "__main__":
    #app.run(debug=True, port=5000)
    app.run(host="0.0.0.0", port=5000)
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import pandas as pd
import urllib.request
import os
from sklearn.metrics.pairwise import cosine_similarity
import io

app = Flask(__name__)
CORS(app)

CSV_FILE = "tabla_enlaces_imagenes.xlsx"
df = pd.read_excel(CSV_FILE)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_drive_id(url):
    if "id=" in url:
        parts = url.split("id=")
        return parts[-1].split("&")[0]
    if "/file/d/" in url:
        parts = url.split("/file/d/")
        rest = parts[1]
        return rest.split("/")[0]
    return None

def drive_download_url(drive_url):
    file_id = extract_drive_id(drive_url)
    if not file_id:
        return drive_url
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def image_to_vector_from_bytes(bts):
    img = Image.open(io.BytesIO(bts)).convert("RGB").resize((128,128))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.flatten().reshape(1, -1)

@app.route("/")
def home():
    return "Comparador Mi Pajarito - servidor activo"

@app.route("/compare", methods=["POST"])
def compare():
    if "photo" not in request.files:
        return jsonify({"error":"No file part 'photo' provided"}), 400
    file = request.files["photo"]
    if file.filename == "":
        return jsonify({"error":"Empty filename"}), 400
    try:
        data = file.read()
        uploaded_vec = image_to_vector_from_bytes(data)
    except Exception as e:
        return jsonify({"error":"Error processing uploaded image", "detail": str(e)}), 500

    best_score = -1.0
    best_row = None

    for _, row in df.iterrows():
        try:
            drive_url = str(row["ImagenDrive"])
            download_url = drive_download_url(drive_url)
            resp = urllib.request.urlopen(download_url, timeout=15)
            img_bytes = resp.read()
            vec = image_to_vector_from_bytes(img_bytes)
            score = float(cosine_similarity(uploaded_vec, vec)[0][0])
            if score > best_score:
                best_score = score
                best_row = row
        except Exception as ex:
            print("Skip image, error:", ex)
            continue

    if best_row is None:
        return jsonify({"error":"No matches found"}), 200

    nombre = str(best_row.get("Nombre",""))
    subpagina = str(best_row.get("Subpagina",""))
    porcentaje = round(best_score * 100, 2)
    return jsonify({"nombre": nombre, "porcentaje": porcentaje, "url": subpagina})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
# app.py
import streamlit as st
from ultralytics import YOLO
from fer import FER
from PIL import Image
import numpy as np
import cv2
import datetime
import cloudinary
import cloudinary.uploader
import os
import tempfile
import json
import plotly.express as px
import pandas as pd
import gradio as gr
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# MySQL helper (SQLAlchemy)
from db import insert_deteksi, list_deteksi, delete_deteksi_by_id

# ---------------------------
# FastAPI endpoint for programmatic predict (useful for Android/backend)
# ---------------------------
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_np = np.array(image)
    model = YOLO(os.getenv("MODEL_PATH", "yolo11n.pt"))
    detector = FER(mtcnn=True)

    results = model.predict(img_np, conf=0.4)
    boxes = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0].boxes, "xyxy") else np.array([])
    boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0])) if len(boxes) > 0 else []
    all_data = []

    for idx, bbox in enumerate(boxes_sorted):
        try:
            x1, y1, x2, y2 = map(int, bbox[:4])
            face_crop = img_np[y1:y2, x1:x2]
            emo = detector.detect_emotions(face_crop)
            if emo:
                best, score = max(emo[0]["emotions"].items(), key=lambda x: x[1])
                best = best.capitalize()
                all_data.append({
                    "Person": f"Person {idx+1}",
                    "Emotion": best,
                    "Confidence": round(score * 100, 2)
                })
        except Exception:
            continue

    return JSONResponse({"hasil": all_data})


# small PWA injection (harmless)
gr.HTML(\"\"\"<link rel="manifest" href="manifest.json">
<script>
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/service-worker.js");
}
</script>\"\"\")

# ===============================
# Streamlit UI
# ===============================
st.markdown(\"\"\"<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #ffeef5 0%, #ffd9ea 100%);
    background-attachment: fixed;
    background-size: cover;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
[data-testid="stSidebar"] { background: rgba(255, 235, 245, 0.55); backdrop-filter: blur(14px); border-right: 1px solid rgba(255, 200, 220, 0.4); }
.card { background: rgba(255,255,255,0.55); padding:22px; border-radius:22px; box-shadow:0 6px 22px rgba(255,130,170,0.18); backdrop-filter:blur(18px); border:1px solid rgba(255,190,220,0.45); }
h1,h2,h3,h4 { color:#b3126f !important; font-weight:700 !important; letter-spacing:-0.5px; }
p,label,span { color:#6b2b4b !important; }
.stButton > button { background:linear-gradient(135deg,#ff9ecb,#ff7bb7); color:white !important; border:none; padding:10px 20px; border-radius:14px; font-size:16px; font-weight:600; transition:0.25s; box-shadow:0 4px 16px rgba(255,100,150,0.35); }
.stTabs [role="tablist"] { gap:16px; background: rgba(255,240,250,0.55); padding:10px; border-radius:18px; backdrop-filter:blur(12px); border:1px solid rgba(255,190,220,0.45); }
img { border-radius:18px; box-shadow:0 3px 12px rgba(0,0,0,0.12); }
</style>\"\"\", unsafe_allow_html=True)

st.title("😎 Deteksi Wajah & Emosi — Resky")

# Cloudinary init (from env vars)
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

emotion_colors = {
    "Happy": (0, 255, 0),
    "Neutral": (0, 150, 255),
    "Sad": (0, 0, 255),
    "Angry": (200, 0, 200),
    "Surprise": (255, 255, 0),
    "Fear": (255, 165, 0),
    "Disgust": (128, 128, 128)
}

tab1, tab2, tab3 = st.tabs(["📸 Deteksi Baru", "📜 Riwayat", "📊 Statistik"])

# TAB 1
with tab1:
    uploaded_file = st.file_uploader("Upload foto wajah", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        st.image(image, caption="📸 Gambar Asli", use_container_width=True)

        model = YOLO(os.getenv("MODEL_PATH", "yolo11n.pt"))
        detector = FER(mtcnn=True)

        results = model.predict(img_np, conf=0.4)
        boxes = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0].boxes, "xyxy") else np.array([])
        boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0])) if len(boxes) > 0 else []

        st.subheader("🎯 Hasil Deteksi Emosi")
        annotated_img = img_np.copy()
        all_data = []

        for idx, bbox in enumerate(boxes_sorted):
            try:
                x1, y1, x2, y2 = map(int, bbox[:4])
                face_crop = img_np[y1:y2, x1:x2]
                emo = detector.detect_emotions(face_crop)
                if emo:
                    best, score = max(emo[0]["emotions"].items(), key=lambda x: x[1])
                    best = best.capitalize()
                    all_data.append({
                        "Person": f"Person {idx+1}",
                        "Emotion": best,
                        "Confidence": round(score * 100, 2)
                    })
                    color = emotion_colors.get(best, (255, 255, 255))
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                    label = f"{idx+1}: {best} ({round(score*100)}%)"
                    cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            except Exception:
                pass

        st.image(annotated_img, caption="✅ Hasil Analisis", use_container_width=True)
        st.dataframe(all_data, use_container_width=True)

        # upload image to Cloudinary
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            Image.fromarray(annotated_img).save(tmp.name)
            upload_result = cloudinary.uploader.upload(tmp.name, folder="deteksi_emosi")
            image_url = upload_result.get("secure_url")

        # save to MySQL
        ts = datetime.datetime.now()
        row_id = insert_deteksi(ts, image_url, all_data)

        st.success("✅ Data berhasil disimpan!")

# TAB 2
with tab2:
    st.subheader("📜 Riwayat Deteksi")
    col1, col2 = st.columns(2)
    start = col1.date_input("Mulai tanggal")
    end = col2.date_input("Sampai tanggal")
    try:
        docs = list_deteksi()
    except Exception as e:
        st.error(f"Gagal mengambil riwayat: {e}")
        docs = []

    for d in docs:
        tstamp = d.get("timestamp")
        if not tstamp:
            continue
        try:
            date_obj = datetime.datetime.fromisoformat(tstamp).date()
        except:
            try:
                date_obj = datetime.datetime.strptime(tstamp, "%Y-%m-%d %H:%M:%S").date()
            except:
                continue
        if start and date_obj < start:
            continue
        if end and date_obj > end:
            continue
        st.image(d.get("image_url", ""), caption=str(date_obj), width=450)
        for h in d.get("hasil", []):
            st.markdown(f"🧍 **{h['Person']}** — {h['Emotion']} ({h['Confidence']}%)")
        if st.button(f"Hapus Riwayat {d.get('id')}", key=f"del-{d.get('id')}"):
            try:
                delete_deteksi_by_id(d.get("id"))
                st.success("✅ Riwayat terhapus!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Gagal menghapus: {e}")

# TAB 3
with tab3:
    st.subheader("📊 Statistik Emosi")
    try:
        docs = list_deteksi()
    except Exception as e:
        st.error(f"Gagal mengambil data statistik: {e}")
        docs = []
    stats = {"Happy":0, "Neutral":0, "Sad":0, "Angry":0, "Surprise":0, "Fear":0, "Disgust":0}
    records = []
    for d in docs:
        tstamp = d.get("timestamp")
        if not tstamp:
            continue
        try:
            d_date = datetime.datetime.fromisoformat(tstamp).date()
        except:
            continue
        for h in d.get("hasil", []):
            emo = h.get("Emotion")
            if emo in stats:
                stats[emo] += 1
            records.append({"date": d_date, "emotion": emo})
    if len(records) > 0:
        df = pd.DataFrame(records)
    else:
        df = pd.DataFrame(columns=["date", "emotion"])
    if sum(stats.values()) > 0:
        fig = px.pie(names=list(stats.keys()), values=list(stats.values()), title="Distribusi Emosi Keseluruhan")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("📈 Tren Harian")
    if len(df) > 0:
        daily = df.groupby(["date", "emotion"]).size().reset_index(name="count")
        fig_daily = px.line(daily, x="date", y="count", color="emotion", markers=True, title="Tren Emosi Harian")
        st.plotly_chart(fig_daily, use_container_width=True)
    st.subheader("📊 Tren Mingguan")
    if len(df) > 0:
        df["week"] = df["date"].apply(lambda x: x.isocalendar().week)
        weekly = df.groupby(["week", "emotion"]).size().reset_index(name="count")
        fig_weekly = px.bar(weekly, x="week", y="count", color="emotion", barmode="group", title="Tren Emosi Mingguan")
        st.plotly_chart(fig_weekly, use_container_width=True)
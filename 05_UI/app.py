import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Bone Fracture Detection",
    layout="centered"
)

st.title("🦴 Bone Fracture Detection System")
st.caption("© SHUBHAM MADDHESIYA")

# ---------------- PATHS ----------------
BASE_PATH = "/Bone_Fracture_Detection_CNN"
MODEL_PATH = f"{BASE_PATH}/03_Models/bone_fracture_model_phase1.h5"
REPORT_DIR = f"{BASE_PATH}/06_Reports"

os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---------------- PDF FUNCTION ----------------
def generate_pdf(result, confidence, save_path):
    c = canvas.Canvas(save_path, pagesize=A4)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2*cm, 28*cm, "Bone Fracture Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(2*cm, 26.5*cm, f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    c.drawString(2*cm, 25.5*cm, f"Prediction Result: {result}")
    c.drawString(2*cm, 24.5*cm, f"Confidence Score: {confidence:.2f}%")

    c.setFont("Helvetica-Oblique", 10)
    c.drawString(2*cm, 3*cm, "Disclaimer: This tool is for educational purposes only.")
    c.drawString(2*cm, 2.3*cm, "© SHUBHAM MADDHESIYA")

    c.save()

# ---------------- UI ----------------
uploaded_file = st.file_uploader(
    "📤 Upload X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    prediction = model.predict(img_array)[0][0]
    confidence = prediction * 100

    if prediction > 0.5:
        result = "Fractured"
    else:
        result = "Normal"

    st.subheader("🧠 Prediction Result")
    st.write(f"**Status:** {result}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    if st.button("📄 Download PDF Report"):
        filename = f"Bone_Fracture_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = os.path.join(REPORT_DIR, filename)
        generate_pdf(result, confidence, pdf_path)
        st.success("✅ PDF Report Generated Successfully!")

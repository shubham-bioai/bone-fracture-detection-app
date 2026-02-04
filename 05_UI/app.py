st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.reportview-container {
    padding: 2rem;
}
</style>
""", unsafe_allow_html=True)


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
BASE_PATH = "."
MODEL_PATH = os.path.join(BASE_PATH, "03_Models", "bone_fracture_model_phase1.h5")
REPORT_DIR = "/tmp/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---------------- PDF FUNCTION ----------------
from io import BytesIO

def generate_pdf(result):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 800, "Bone Fracture Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 760, f"Prediction Result: {result}")
    c.drawString(50, 730, f"Generated on: {datetime.now().strftime('%d %B %Y %H:%M')}")
    c.drawString(50, 700, "Disclaimer: This is an AI-based assessment.")
    c.drawString(50, 670, "© SHUBHAM MADDHESIYA")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

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

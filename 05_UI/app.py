# ==============================
# Bone Fracture Detection App
# © SHUBHAM MADDHESIYA
# ==============================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🦴",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main-card {
    padding: 20px;
    border-radius: 12px;
    background-color: #f9f9f9;
    box-shadow: 0 0 10px rgba(0,0,0,0.05);
}
.result-card {
    padding: 15px;
    border-radius: 10px;
    background-color: #ffffff;
    border-left: 6px solid #2b7cff;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 13px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>🦴 Bone Fracture Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered X-ray analysis for fracture detection</p>", unsafe_allow_html=True)

# ---------------- MODEL LOAD ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("03_Models/bone_fracture_model_phase1.h5")

model = load_model()

# ---------------- IMAGE UPLOAD ----------------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Upload X-ray Image",
    type=["jpg", "jpeg", "png"]
)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PDF GENERATOR ----------------
def generate_pdf(result, confidence):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, 800, "Bone Fracture Detection Report")

    # Patient Summary
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 760, "Patient Summary")

    c.setFont("Helvetica", 12)
    c.drawString(50, 735, f"Prediction Result: {result}")
    c.drawString(50, 715, f"Confidence Score: {confidence:.2f}%")
    c.drawString(50, 695, f"Report Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}")

    # Clinical Info
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 655, "Clinical Notes")

    c.setFont("Helvetica", 11)
    c.drawString(50, 630, "- This result is generated using a deep learning model.")
    c.drawString(50, 610, "- The system analyzes X-ray images for fracture patterns.")
    c.drawString(50, 590, "- This is NOT a medical diagnosis.")

    # Disclaimer
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 550, "Disclaimer: This AI-based assessment is for educational and")
    c.drawString(50, 535, "screening purposes only. Consult a certified medical professional.")

    # Footer
    c.setFont("Helvetica", 10)
    c.drawString(50, 500, "© SHUBHAM MADDHESIYA | Bone Fracture Detection System")

    c.showPage()
    c.save()

    buffer.seek(0)
    return buffer

# ---------------- PREDICTION ----------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Predict
THRESHOLD = 0.4
result = "Fractured" if prediction > THRESHOLD else "Normal"
confidence = prediction if prediction > THRESHOLD else 1 - prediction

    # Result UI
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.subheader("🧾 Prediction Result")
    st.write(f"**Status:** {result}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    # PDF Download
    pdf_buffer = generate_pdf(result, confidence)

    st.download_button(
        label="📄 Download Medical PDF Report",
        data=pdf_buffer,
        file_name="Bone_Fracture_Report.pdf",
        mime="application/pdf"
    )

# ---------------- FOOTER ----------------
st.markdown("<div class='footer'>© SHUBHAM MADDHESIYA · AI for Healthcare</div>", unsafe_allow_html=True)

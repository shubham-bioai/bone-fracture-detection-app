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

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center;'>🦴 Bone Fracture Detection System</h1>
    <p style='text-align: center; color: grey;'>
    AI-powered X-ray analysis for fracture detection
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>© SHUBHAM MADDHESIYA</p>",
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "03_Models/bone_fracture_model_phase1.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------- PDF FUNCTION ----------------
def generate_pdf(result, confidence):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 800, "Bone Fracture Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 760, f"Prediction Result: {result}")
    c.drawString(50, 730, f"Confidence Score: {confidence:.2f}%")
    c.drawString(50, 700, f"Generated on: {datetime.now().strftime('%d %B %Y %H:%M')}")

    c.drawString(50, 660, "Disclaimer:")
    c.drawString(50, 640, "This is an AI-based assessment and not a medical diagnosis.")

    c.drawString(50, 600, "© SHUBHAM MADDHESIYA")

    c.showPage()
    c.save()

    buffer.seek(0)
    return buffer

# ---------------- UPLOAD SECTION ----------------
st.markdown("### 📤 Upload X-ray Image")

uploaded_file = st.file_uploader(
    "Supported formats: JPG, PNG, JPEG",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("### 🖼 Uploaded Image")
    st.image(image, use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        result = "Fractured"
        confidence = prediction * 100
        st.error(f"🩺 Result: {result}")
    else:
        result = "Normal"
        confidence = (1 - prediction) * 100
        st.success(f"✅ Result: {result}")

    st.info(f"Confidence Score: {confidence:.2f}%")

    # ---------------- PDF DOWNLOAD ----------------
    st.markdown("### 📄 Report")

    if st.button("Generate PDF Report"):
        pdf_buffer = generate_pdf(result, confidence)

        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_buffer,
            file_name="Bone_Fracture_Report.pdf",
            mime="application/pdf"
        )

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; color: grey; font-size: 13px;'>
    Built with Streamlit & Deep Learning
    </p>
    """,
    unsafe_allow_html=True
)

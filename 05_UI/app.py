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
body {
    background-color: #f5f7fa;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.title {
    text-align: center;
    font-size: 34px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: grey;
    margin-bottom: 30px;
}

.footer {
    text-align: center;
    color: grey;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🦴 Bone Fracture Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered X-ray analysis</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>© SHUBHAM MADDHESIYA</div>", unsafe_allow_html=True)

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
    c.drawString(50, 640, "This AI result is not a medical diagnosis.")

    c.drawString(50, 600, "© SHUBHAM MADDHESIYA")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ---------------- UPLOAD CARD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 📤 Upload X-ray Image")

uploaded_file = st.file_uploader(
    "Supported formats: JPG, PNG, JPEG",
    type=["jpg", "png", "jpeg"]
)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PROCESS ----------------
if uploaded_file:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    image = Image.open(uploaded_file).convert("RGB")
    st.markdown("### 🖼 Uploaded X-ray")
    st.image(image, use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

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

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- PDF CARD ----------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📄 Download Medical Report")

    if st.button("Generate PDF Report"):
        pdf_buffer = generate_pdf(result, confidence)

        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_buffer,
            file_name="Bone_Fracture_Report.pdf",
            mime="application/pdf"
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='footer'>Built with Deep Learning & Streamlit</div>", unsafe_allow_html=True)

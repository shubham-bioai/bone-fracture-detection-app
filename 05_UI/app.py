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

# ---------------- PROFESSIONAL PDF GENERATOR ----------------
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

def generate_pdf(result, confidence):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # ---------------- HEADER ----------------
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 60, "Bone Fracture Detection Report")

    c.setFont("Helvetica", 11)
    c.drawString(50, height - 90, "AI-Assisted X-ray Analysis")
    c.drawRightString(width - 50, height - 90, datetime.now().strftime("%d %B %Y, %H:%M"))

    c.line(50, height - 100, width - 50, height - 100)

    # ---------------- PATIENT SUMMARY ----------------
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 140, "1. Examination Summary")

    c.setFont("Helvetica", 12)
    c.drawString(60, height - 170, f"• Predicted Condition: {result}")
    c.drawString(60, height - 195, f"• Model Confidence: {confidence:.2f}%")
    c.drawString(60, height - 220, "• Imaging Type: X-ray")

    # ---------------- FINDINGS ----------------
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 260, "2. AI Findings")

    c.setFont("Helvetica", 12)
    if result.lower() == "fractured":
        c.drawString(60, height - 290, 
            "• The AI model detected patterns consistent with a bone fracture.")
        c.drawString(60, height - 315, 
            "• Structural discontinuity and abnormal bone texture were observed.")
    else:
        c.drawString(60, height - 290, 
            "• No significant fracture patterns were detected by the AI model.")
        c.drawString(60, height - 315, 
            "• Bone structure appears within normal limits.")

    # ---------------- CLINICAL GUIDANCE ----------------
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 355, "3. Suggested Clinical Guidance")

    c.setFont("Helvetica", 12)
    c.drawString(60, height - 385, 
        "• Consult a certified orthopedic specialist for clinical confirmation.")
    c.drawString(60, height - 410, 
        "• Further imaging (X-ray / CT / MRI) may be required if symptoms persist.")
    c.drawString(60, height - 435, 
        "• Do not rely solely on this report for medical decisions.")

    # ---------------- DISCLAIMER ----------------
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 475, "4. Disclaimer")

    c.setFont("Helvetica", 11)
    c.drawString(60, height - 505, 
        "This report is generated using an AI model and is intended for")
    c.drawString(60, height - 525, 
        "educational and assistive purposes only. It is NOT a medical diagnosis.")

    # ---------------- FOOTER ----------------
    c.line(50, 90, width - 50, 90)
    c.setFont("Helvetica", 10)
    c.drawString(50, 70, "© SHUBHAM MADDHESIYA")
    c.drawRightString(width - 50, 70, "Bone Fracture Detection System")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

#--------------Download-----------

pdf_buffer = generate_pdf(result, confidence)
st.download_button(
    label="📄 Download Detailed PDF Report",
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

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
from pathlib import Path
import shutil

# ------------------------------
# Load YOLOv8 model
# ------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # You can change to yolov8s.pt for better accuracy

model = load_model()

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="YOLO Object Recognition", layout="wide")
st.title("🖼️🔎 YOLOv8 Object Recognition App")
st.write("Upload one or more images, and I'll detect objects in them!")

uploaded_files = st.file_uploader(
    "📂 Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Display original image
        st.subheader(f"Original Image: {uploaded_file.name}")
        st.image(Image.open(temp_path), caption="Uploaded Image", use_column_width=True)

        # Run YOLO detection
        st.info("🔄 Running object detection… please wait.")
        results = model.predict(source=temp_path, save=True)

        # Get first result
        result = results[0]
        save_dir = Path(result.save_dir)

        # Try to find the saved annotated image
        predicted_image_path = save_dir / uploaded_file.name
        if not predicted_image_path.exists():
            predicted_image_path = next(save_dir.glob("*.jpg"), None)

        # Show detected image
        st.subheader("✅ Detection Results")
        if predicted_image_path and predicted_image_path.exists():
            st.image(str(predicted_image_path), caption="Detected Objects", use_column_width=True)
        else:
            st.warning("⚠️ Annotated image not found.")

        # List detected objects
        st.subheader("📋 Detected Objects")
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            st.write("❌ No objects detected.")
        else:
            for box in boxes:
                cls_id = int(box.cls.item())  # Use .item() to extract scalar from tensor
                label = result.names[cls_id]
                conf = float(box.conf.item())
                st.write(f"- **{label}** with confidence `{conf:.2f}`")

        # Clean up temp directory
        shutil.rmtree(temp_dir)

st.markdown("---")
st.caption("Built with ❤️ using YOLOv8 and Streamlit")

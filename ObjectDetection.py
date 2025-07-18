import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# =========================
# Load YOLOv8 model (nano version, change to yolov8s.pt for more accuracy)
# =========================
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="YOLO Object Recognition", layout="wide")
st.title("🖼️🔎 YOLOv8 Object Recognition App")
st.write("Upload one or more images, and I'll detect objects in them using YOLOv8!")

uploaded_files = st.file_uploader(
    "Choose image files", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save uploaded file to a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Display original image
        st.subheader(f"Original Image: {uploaded_file.name}")
        st.image(Image.open(temp_path), caption="Uploaded Image", use_column_width=True)

        # Run YOLOv8 detection
        st.info("🔄 Running object detection...")
        results = model(temp_path, save=True)

        # Get detection results
        result = results[0]
        detected_image_path = result.save_dir / uploaded_file.name  # saved in runs/detect/predict

        # Show detected image
        st.subheader("✅ Detection Results")
        st.image(str(detected_image_path), caption="Detected Objects", use_column_width=True)

        # List detected objects
        st.subheader("📋 Detected Objects")
        if len(result.boxes) == 0:
            st.write("❌ No objects detected.")
        else:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                conf = float(box.conf[0])
                st.write(f"- **{label}** with confidence `{conf:.2f}`")

st.markdown("---")
st.caption("Built with ❤️ using YOLOv8 and Streamlit")

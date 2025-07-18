import streamlit as st
import os
import tempfile
import cv2
import torch
import open_clip
import faiss
import numpy as np
from PIL import Image
from pathlib import Path

# ---------------------------
# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e32')
model = model.to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# ---------------------------
# Helper: extract frames from a video
def extract_frames(video_path, fps_extract=1):
    frames = []
    timestamps = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if int(frame_num % max(int(fps / fps_extract),1)) == 0:
            # save frame to memory
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img_rgb))
            timestamps.append(frame_num / fps)
        frame_num += 1
    cap.release()
    return frames, timestamps

# ---------------------------
# Helper: embed images
def embed_images(images):
    all_feats = []
    with torch.no_grad():
        for img in images:
            img_prep = preprocess(img).unsqueeze(0).to(device)
            feat = model.encode_image(img_prep)
            feat /= feat.norm(dim=-1, keepdim=True)
            all_feats.append(feat.cpu().numpy())
    return np.vstack(all_feats)

# ---------------------------
# Streamlit UI
st.title("🔎 Media Search with AI")

uploaded_files = st.file_uploader("Upload Images or Videos", type=["jpg","jpeg","png","mp4","avi","mov"], accept_multiple_files=True)
query = st.text_input("Enter a text to search (e.g., 'car', 'road', 'wind turbine'):")

if st.button("Run Search"):
    if not uploaded_files or not query:
        st.error("Please upload files and enter a search query.")
    else:
        st.info("🔄 Processing files… this might take a moment.")
        # Temporary folder
        temp_dir = tempfile.mkdtemp()
        embeddings = []
        metadata = []  # (filename, timestamp/frame_idx)
        for f in uploaded_files:
            file_path = os.path.join(temp_dir, f.name)
            with open(file_path, "wb") as out:
                out.write(f.read())
            if f.name.lower().endswith((".jpg",".jpeg",".png")):
                img = Image.open(file_path).convert("RGB")
                embeddings.append(embed_images([img]))
                metadata.append((f.name, None))
            else:
                # It's a video
                frames, times = extract_frames(file_path)
                if frames:
                    feats = embed_images(frames)
                    embeddings.append(feats)
                    for t in times:
                        metadata.append((f.name, t))
        if embeddings:
            embeddings = np.vstack(embeddings)
        else:
            st.warning("No valid media files processed.")
            st.stop()

        # Build FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        # Query
        text_tokens = tokenizer([query]).to(device)
        with torch.no_grad():
            text_feat = model.encode_text(text_tokens)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
        text_np = text_feat.cpu().numpy()
        D, I = index.search(text_np, k=5)

        # Show results
        st.subheader("✅ Top Matches")
        for rank, idx in enumerate(I[0]):
            fname, ts = metadata[idx]
            st.write(f"**{rank+1}. File:** {fname}")
            if ts is not None:
                st.write(f"⏱ Timestamp: {ts:.2f} seconds")
            # Display image or frame preview
            if fname.lower().endswith((".jpg",".jpeg",".png")):
                st.image(os.path.join(temp_dir, fname), caption=fname)
            else:
                # For videos, re-extract that frame
                vidcap = cv2.VideoCapture(os.path.join(temp_dir, fname))
                fps = vidcap.get(cv2.CAP_PROP_FPS)
                frame_number = int(ts * fps)
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = vidcap.read()
                vidcap.release()
                if ret:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, caption=f"{fname} @ {ts:.2f}s")

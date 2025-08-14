import streamlit as st
import numpy as np
import cv2
from src.model import TextGrounder
from src.utils import draw_predictions, box_to_int

st.set_page_config(page_title="Scene Localization via Text", layout="wide")

st.title("Scene Localization in Dense Images via Natural Language Queries")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Score threshold", 0.0, 1.0, 0.25, 0.01)
    topk = st.number_input("Top-k", min_value=1, max_value=20, value=5)
    run_btn = st.button("Run")

query = st.text_input("Enter your query (comma-separated for multiple):", "a person talking, two people fighting")
img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if "model" not in st.session_state:
    st.session_state.model = None

if img_file:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Failed to decode image.")
    else:
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        st.image(image, caption="Input Image", use_column_width=True)

    if run_btn:
        if st.session_state.model is None:
            with st.spinner("Loading zero-shot detector (first time can take a while)..."):
                st.session_state.model = TextGrounder()
        model = st.session_state.model

        queries = [q.strip() for q in query.split(",") if q.strip()]
        with st.spinner("Running inference..."):
            preds = model.predict(image, queries, score_threshold=threshold, topk=topk)

        if preds:
            best = max(preds, key=lambda p: p["score"]) 
            crop = model.crop_from_box(image, best["box"]) 
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Detections")
                st.image(draw_predictions(image, preds), use_column_width=True)
            with col2:
                st.subheader(f"Top match: {best['label']} ({best['score']:.2f}) @ {box_to_int(best['box'])}")
                st.image(crop, use_column_width=True)
        else:
            st.info("No detections above threshold. Try adjusting the threshold or query.")
else:
    st.info("Upload an image to get started.")

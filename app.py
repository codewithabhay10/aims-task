import streamlit as st
import numpy as np
import cv2
from src.model import TextGrounder
from src.utils import draw_predictions, box_to_int

st.set_page_config(page_title="Scene Localization via Text", layout="wide")

st.title("Scene Localization in Dense Images via Natural Language Queries")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Score threshold", 0.0, 1.0, 0.08, 0.01, help="Lower to include weaker raw candidates (useful for debugging)")
    topk = st.number_input("Top-k", min_value=1, max_value=50, value=20)
    st.markdown("---")
    auto_select = st.checkbox("Auto-select (adaptive)", value=False, help="Enable adaptive thresholding (knee or ratio) to pick likely candidates when absolute scores are low")
    auto_mode = st.radio("Auto mode", options=["knee", "ratio"], index=0, help="knee: find score knee; ratio: top_score * auto_ratio")
    auto_ratio = st.slider("Auto ratio", 0.1, 1.0, 0.6, 0.05, help="Multiplier applied to top score when using ratio mode")
    min_keep = st.number_input("Min keep when auto-selecting", min_value=1, max_value=50, value=1)
    st.markdown("---")
    nms_iou = st.slider("NMS IoU", 0.1, 0.9, 0.5, 0.05, help="Suppress overlapping boxes per label")
    min_box_pct = st.slider("Min box area (% of image)", 0.0, 10.0, 1.0, 0.1, help="Drop tiny boxes below this percent of the image area")
    per_query_best = st.checkbox("One best box per query", value=True)
    st.markdown("---")
    size_bias = st.slider("Size bias for top match (gamma)", 0.0, 2.0, 0.3, 0.05, help="0=no bias, higher prefers larger boxes when picking the top match")
    crop_margin_pct = st.slider("Crop margin (%)", 0.0, 20.0, 5.0, 0.5, help="Expand crop around the chosen box for context")
    run_btn = st.button("Run")

query = st.text_input("Enter your query (comma-separated for multiple):", "a person talking, two people fighting")
img_files = st.file_uploader("Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if "model" not in st.session_state:
    st.session_state.model = None

if img_files:
    st.write(f"{len(img_files)} image(s) selected.")
    if run_btn:
        if st.session_state.model is None:
            with st.spinner("Loading zero-shot detector (first time can take a while)..."):
                st.session_state.model = TextGrounder()
        model = st.session_state.model

        queries = [q.strip() for q in query.split(",") if q.strip()]
        if not queries:
            st.warning("Please enter at least one query.")
        else:
            for i, img_file in enumerate(img_files, start=1):
                file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
                bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if bgr is None:
                    st.error(f"[{i}] Failed to decode image: {img_file.name}")
                    continue
                image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                with st.spinner(f"Running inference on {img_file.name} ({i}/{len(img_files)})..."):
                    preds = model.predict(
                        image,
                        queries,
                        score_threshold=threshold,
                        topk=topk,
                        auto_select=auto_select,
                        auto_ratio=auto_ratio,
                        min_keep=min_keep,
                        auto_mode=auto_mode,
                        iou_threshold=nms_iou,
                        min_box_rel_area=(min_box_pct/100.0),
                        per_query_best=per_query_best,
                    )

                st.markdown(f"### {img_file.name}")
                if preds:
                    # size-biased top match selection
                    H, W = image.shape[:2]
                    img_area = float(H * W)
                    def eff_score(p):
                        x1, y1, x2, y2 = p["box"]
                        area = max(0.0, (x2 - x1) * (y2 - y1))
                        area_norm = max(1e-6, min(1.0, area / (img_area + 1e-9)))
                        return float(p["score"]) * (area_norm ** size_bias)
                    best = max(preds, key=eff_score)
                    crop = model.crop_from_box(image, best["box"], margin_ratio=(crop_margin_pct/100.0)) 
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Detections")
                        st.image(draw_predictions(image, preds), use_container_width=True)
                    with col2:
                        st.subheader(f"Top match: {best['label']} ({best['score']:.2f}) @ {box_to_int(best['box'])}")
                        st.image(crop, use_container_width=True)
                else:
                        st.info("No detections above threshold. Showing raw model outputs for debugging.")
                        # show raw outputs
                        raw = model.get_raw_outputs(image, queries)
                        st.write(f"Raw candidates: {len(raw['scores'])}")
                        if len(raw['scores']) > 0:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(1, 1, figsize=(4, 1.2))
                            ax.hist(raw['scores'], bins=20)
                            ax.set_xlabel('score')
                            ax.set_title('Raw score distribution')
                            st.pyplot(fig)

                            # draw raw top-5 boxes
                            order = (-raw['scores']).argsort()[:5]
                            boxes = raw['boxes'][order]
                            scores = raw['scores'][order]
                            labels_raw = raw['labels'][order]
                            # map label indices back to query text when possible
                            vis_raw = draw_predictions(
                                image.copy(),
                                [
                                    {
                                        'box': tuple(b),
                                        'score': float(s),
                                        'label': (queries[int(l)] if 0 <= int(l) < len(queries) else str(int(l)))
                                    }
                                    for b, s, l in zip(boxes, scores, labels_raw)
                                ]
                            )
                            st.image(vis_raw, caption='Top-5 raw candidates')
                        else:
                            st.write('Model returned zero candidates.')
else:
    st.info("Upload an image to get started.")

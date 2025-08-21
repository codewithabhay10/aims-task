# Scene Localization in Dense Images via Natural Language Queries

End‑to‑end prototype to ground free‑form text queries in dense images and return bounding boxes and crops. Built on top of open‑vocabulary detection (OWLv2) from Hugging Face Transformers, with OpenCV/NumPy for image I/O and Streamlit for an interactive UI. Includes a CLI, batch runner, adaptive post‑processing, and an optional CLIP‑based reranker.

## Contents
- What’s included
- Setup and environment
- How it works (architecture)
- Streamlit UI guide
- CLI and batch usage
- Post‑processing parameters (and when to tweak them)
- Methods tried and iterations (failures and fixes)
- Known limitations and practical tips
- Troubleshooting
- Next steps

---

## What’s included
- Core model wrapper: `src/model.py` (OWLv2, robust post‑processing)
- Utilities: `src/utils.py` (drawing, box helpers)
- Optional re‑ranker: `src/reranker.py` (CLIP crop re‑ranking)
- Single‑image CLI: `src/infer.py`
- Batch CLI: `src/batch_infer.py`
- Streamlit app: `app.py`
- Requirements and README

---

## Setup and environment
Recommended Python: 3.10–3.12 on Windows. Create a venv and install deps.

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Torch: for CPU
```cmd
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

GPU users should install a CUDA‑matching torch build from pytorch.org for speed.

Run the Streamlit app from this venv:
```cmd
python -m streamlit run app.py
```

Notes
- First run downloads OWLv2 weights (~1GB).
- We set the processor to `use_fast=True` for speed and to silence warnings.
- The code uses the new grounded post‑process when available, falling back to the old API.

---

## How it works (architecture)
1. Preprocess: OWLv2 processor encodes the image and a list of text queries.
2. Model: OWLv2 predicts candidate boxes and scores per query (open‑vocabulary detection).
3. Post‑processing (our pipeline in `TextGrounder.predict`):
	 - Clip boxes to image bounds; drop degenerate boxes.
	 - Filter by minimum relative area (`min_box_rel_area`); optional maximum area.
	 - Adaptive selection (optional): choose a dynamic threshold per image
		 - ratio: threshold = top_score × `auto_ratio`
		 - knee: threshold at largest drop in sorted scores; guarantees at least `min_keep` boxes
	 - Per‑label NMS with configurable IoU.
	 - Optional “one best per query”.
	 - Final Top‑K.
4. Optional re‑ranking: CLIP similarity on the crop with the query and mix with detector score.
5. Display: draw boxes, compute a size‑biased “top match”, crop with optional margin.

---

## Streamlit UI guide
Sidebar controls:
- Score threshold: base filter; keep low (0.05–0.12) when using Auto‑select.
- Top‑k: number of candidates to keep after filtering.
- Auto‑select (adaptive): enables ratio/knee thresholding; knee is a good default.
- Auto ratio: used in ratio mode; 0.6–0.8 typical.
- Min keep: enforce at least this many candidates in Auto‑select.
- NMS IoU: 0.6–0.7 reduces overlapping small boxes.
- Min box area (%): drop tiny boxes (<1–3%) that often cause false positives.
- One best per query: keep the best candidate per query string.
- Size bias (gamma): bias top match toward larger boxes (0.4–1.0 typical).
- Crop margin (%): expand crop around the chosen box (5–10% makes nicer crops).
- Re‑rank with CLIP: reorder using crop–text similarity; `alpha` mixes detector vs CLIP.

If no detections pass filters, the app shows a histogram of raw scores and top‑5 raw boxes for debugging.

---

## CLI usage
Single image:
```cmd
python -m src.infer assets\sample.jpg "a man snatching a chain" --threshold 0.25 --topk 5 --crop_out outputs\crop.jpg --vis_out outputs\vis.jpg
```

Multiple queries:
```cmd
python -m src.infer assets\market.jpg "a vendor selling vegetables" "people talking"
```

### Batch CLI
Process folders and files and save outputs to `outputs/`:
```cmd
python -m src.batch_infer assets\ images\more_images --queries "a vendor selling vegetables" "people talking" --threshold 0.25 --topk 5 --out outputs
```

---

## Post‑processing parameters (and when to tweak)
- Nothing shows: lower Score threshold (0.05–0.1), enable Auto‑select (knee), ensure Top‑k ≥ 20.
- Too many tiny boxes: increase Min box area to 1–3%, raise NMS IoU to 0.6–0.7.
- Model prefers small part (e.g., “book” only): increase Size bias (0.8–1.2), Min box area to 3–6%, enable One best per query.
- Wrong ordering of plausible boxes: enable Re‑rank with CLIP; alpha 0.6–0.8.
- Boxes too tight: increase Crop margin to 5–10%.

---

## Methods tried and iterations (failures and fixes)

1) Baseline zero‑shot OWLv2 with fixed threshold
- Symptom: lots of overlapping small boxes; “no detections” if threshold too high.
- Fixes: added Top‑K, area filter, per‑label NMS.

2) Windows import issues
- Pillow C‑extension error → removed Pillow; switched to OpenCV.
- NumPy binary mismatch → pinned `numpy==2.1.3`.
- `transformers` missing in Streamlit → ensured the app is launched from project venv.

3) Degenerate/negative boxes
- Symptom: invalid crops or negative coordinates.
- Fix: clip boxes to image bounds, drop degenerate boxes.

4) Adaptive selection
- Implemented ratio and knee modes with `min_keep` to reduce manual tuning.
- UI wiring for Auto‑select + controls.

5) “Book instead of person holding a book”
- Symptom: detector focuses on the most discriminative part.
- Fixes: size‑biased top match; min box area; higher NMS IoU; query phrasing (“full person holding a book”).

6) Re‑ranking with CLIP (optional)
- Added `src/reranker.py` and UI toggle to combine detector and CLIP crop similarity.
- Improves semantic alignment of the top selection in many cases.

7) API deprecations and warnings
- Switched to grounded post‑process if available; set processor `use_fast=True`.

8) Debugging “no detections”
- Added raw output panel in the UI: score histogram + top‑5 raw boxes.

---
## Known limitations and tips
- Zero‑shot open‑vocabulary detection may yield low absolute scores; use Auto‑select.
- Distractors and tiny objects are hard; increase min box area or try re‑ranking.
- Query phrasing matters. Provide variants: “person wearing a mask, masked face, face mask”.
- For consistent, domain‑specific accuracy, consider collecting 50–200 labeled examples per intent and fine‑tuning or training a re‑ranker on your own data.

---

## Troubleshooting
- ModuleNotFoundError: transformers — ensure Streamlit runs from the project venv.
- NumPy/Pillow import errors on Windows — use OpenCV for I/O; pin `numpy==2.1.3`.
- “No detections” even at low threshold — enable Auto‑select (knee), raise Top‑k; inspect raw histogram.
- Deprecation warnings (post_process_object_detection) — handled via grounded post‑process fallback.

---

## Next steps
- Add composition queries (e.g., person holding X): return person box only if it overlaps an X box.
- Multi‑scale inference for small objects.
- Lightweight fine‑tuning or a learned crop re‑ranker on your dataset.

---

## License
For research/demonstration use. Check each dependency’s license.

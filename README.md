# Scene Localization in Dense Images via Natural Language Queries

This is a simple, runnable prototype that performs zero-shot text-conditioned object detection and returns relevant crops from dense images. It uses OWLv2 (Open-Vocabulary detection) from Hugging Face Transformers. Images are processed with NumPy + OpenCV (no Pillow dependency).

## Features
- Input: image + text query (single or multiple, comma-separated)
- Output: bounding boxes and a best-match crop
- CLI for batch/automation
- Streamlit app for a quick interactive demo

## Install
Create a virtual environment (optional) and install deps. First run may download model weights (~1GB). If using Python 3.13, some binary wheels may lag; Python 3.10–3.12 is recommended.

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you have a compatible GPU with CUDA, install the matching `torch` build from pytorch.org for best speed. For CPU-only:

```cmd
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## CLI usage
```cmd
python -m src.infer assets/sample.jpg "a man snatching a chain" --threshold 0.25 --topk 5 --crop_out outputs/crop.jpg --vis_out outputs/vis.jpg
```
Note: You can pass multiple queries separated by spaces, or quote each one:
```cmd
python -m src.infer assets/market.jpg "a vendor selling vegetables" "people talking"
```

## Streamlit app
```cmd
streamlit run app.py
```
- Upload an image.
- Enter one or more queries, comma-separated.
- Click Run. You’ll see detections and the best crop.

## Data and training
This prototype is zero-shot (no training required). For a full project, build a dataset of dense scenes with text annotations and fine-tune an open-vocabulary grounding model (e.g., GroundingDINO, OWLv2 finetuning via detection heads). Consider hard negatives and multi-instance scenes.

## Notes and tips
- Queries matter: write specific phrases (e.g., "two men arguing" vs. "people"). Try several phrasings.
- Adjust threshold: lower it to see more candidates when nothing shows; raise it to filter noise.
- Very dense or tiny objects may require higher-resolution inputs; try the full-res image.

## Deliverables checklist
- Working prototype (CLI + Streamlit) — done
- Returns cropped region — done
- Documentation — this README
- Demo video — create a short screen recording of the Streamlit app with two different queries

## License
For research/demonstration use. Check each dependency’s license.

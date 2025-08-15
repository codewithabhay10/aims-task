import torch
import numpy as np
from typing import List, Dict, Tuple
from transformers import CLIPProcessor, CLIPModel


class ClipReranker:
    """Re-rank detection candidates using CLIP text-image similarity.

    final_score = alpha * det_score_norm + (1 - alpha) * clip_sim_norm
    Stores both clip and mixed scores in each prediction.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()

    @staticmethod
    def _crop(image: np.ndarray, box: Tuple[float, float, float, float]) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))
        return image[y1:y2, x1:x2]

    @torch.no_grad()
    def rerank(self, image: np.ndarray, preds: List[Dict], query: str, alpha: float = 0.7) -> List[Dict]:
        if not preds:
            return preds
        crops = [self._crop(image, p["box"]) for p in preds]
        # Replace empty crops with tiny center patch
        h, w = image.shape[:2]
        safe_crops = []
        for c in crops:
            if c.size == 0 or c.shape[0] < 2 or c.shape[1] < 2:
                cy, cx = h // 2, w // 2
                safe_crops.append(image[max(0, cy - 1):min(h, cy + 1), max(0, cx - 1):min(w, cx + 1)])
            else:
                safe_crops.append(c)

        inputs = self.processor(text=[query], images=safe_crops, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        sims = out.logits_per_image.squeeze(-1).detach().cpu().numpy()  # (N,)
        # Normalize similarities 0..1
        sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)

        det_scores = np.array([float(p["score"]) for p in preds])
        # Normalize det scores 0..1
        if det_scores.size > 0:
            dmin, dmax = det_scores.min(), det_scores.max()
            if dmax > dmin:
                det_norm = (det_scores - dmin) / (dmax - dmin)
            else:
                det_norm = np.zeros_like(det_scores)
        else:
            det_norm = det_scores

        mixed = alpha * det_norm + (1.0 - alpha) * sims
        order = (-mixed).argsort()

        out_preds: List[Dict] = []
        for idx in order:
            p = dict(preds[idx])
            p["score_clip"] = float(sims[idx])
            p["score_mixed"] = float(mixed[idx])
            out_preds.append(p)
        return out_preds

import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from typing import List, Tuple, Optional, Dict
import numpy as np


class TextGrounder:
    """
    Zero-shot text-conditioned detection using OWLv2.

    Contract:
    - Input: image as numpy array in RGB (H, W, 3), list[str] queries, score_threshold float
    - Output: list of dicts: {"box": (x1,y1,x2,y2), "score": float, "label": str}
    - Errors: raises RuntimeError on model/processor load failure
    """

    def __init__(self, device: Optional[str] = None, model_name: str = "google/owlv2-base-patch16"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.processor = Owlv2Processor.from_pretrained(model_name)
            self.model = Owlv2ForObjectDetection.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

    @torch.no_grad()
    def predict(self, image: np.ndarray, queries: List[str], score_threshold: float = 0.2,
                topk: int = 10) -> List[Dict]:
        # Expect image in RGB uint8 [H,W,3]
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be an RGB array of shape (H,W,3)")
        inputs = self.processor(text=[queries], images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        h, w = image.shape[:2]
        target_sizes = torch.tensor([(h, w)]).to(self.device)  # (H, W)
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)
        result = results[0]

        boxes = result["boxes"].cpu().numpy()  # xyxy in pixels
        scores = result["scores"].cpu().numpy()
        labels = result["labels"].cpu().numpy()

        keep = scores >= score_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # Top-k by score
        if len(scores) > topk:
            idx = np.argsort(-scores)[:topk]
            boxes, scores, labels = boxes[idx], scores[idx], labels[idx]

        predictions = []
        for box, score, label_idx in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [float(v) for v in box]
            label = queries[label_idx] if 0 <= label_idx < len(queries) else str(label_idx)
            predictions.append({
                "box": (x1, y1, x2, y2),
                "score": float(score),
                "label": label,
            })
        return predictions

    @staticmethod
    def crop_from_box(image: np.ndarray, box: Tuple[float, float, float, float]) -> np.ndarray:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = map(lambda v: int(max(0, v)), [x1, y1, x2, y2])
        h, w = image.shape[:2]
        x2 = min(w, x2)
        y2 = min(h, y2)
        return image[y1:y2, x1:x2]

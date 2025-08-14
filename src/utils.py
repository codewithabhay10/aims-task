from typing import List, Dict, Tuple
import numpy as np
import cv2


def draw_predictions(image: np.ndarray, preds: List[Dict], show_labels: bool = True) -> np.ndarray:
    vis = image.copy()
    for p in preds:
        x1, y1, x2, y2 = [int(round(v)) for v in p["box"]]
        score = p.get("score", 0.0)
        label = p.get("label", "")
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if show_labels:
            txt = f"{label} {score:.2f}" if label else f"{score:.2f}"
            ((tw, th), _) = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0, 0, 255), -1)
            cv2.putText(vis, txt, (x1 + 3, max(0, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def box_to_int(box: Tuple[float, float, float, float]):
    x1, y1, x2, y2 = box
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))

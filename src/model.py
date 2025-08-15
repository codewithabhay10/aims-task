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
            # Prefer fast image processor to avoid the warning and get minor speedups
            self.processor = Owlv2Processor.from_pretrained(model_name, use_fast=True)
            self.model = Owlv2ForObjectDetection.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

    @torch.no_grad()
    def predict(self,
                image: np.ndarray,
                queries: List[str],
                score_threshold: float = 0.2,
                topk: int = 10,
                apply_nms: bool = True,
                iou_threshold: float = 0.5,
                min_box_rel_area: float = 0.0,
                max_box_rel_area: Optional[float] = None,
                auto_select: bool = False,
                auto_ratio: float = 0.6,
                min_keep: int = 1,
                per_query_best: bool = False,
                auto_mode: str = "ratio") -> List[Dict]:
        """Run zero-shot detection and apply robust post-processing.

        Args:
            image: RGB numpy array (H,W,3)
            queries: list of query strings
            score_threshold: minimum score to keep (if auto_select True this may be overridden)
            topk: maximum number of predictions to return
            apply_nms: whether to run per-label NMS
            iou_threshold: IoU for NMS
            min_box_rel_area: filter boxes smaller than this fraction of image area
            auto_select: if True use adaptive selection (ratio or knee) to set a dynamic threshold
            auto_ratio: multiplier of top score when using ratio auto mode
            min_keep: minimum number of boxes to keep when auto-selecting
            per_query_best: keep only the best box per query label
            auto_mode: 'ratio' or 'knee'

        Returns:
            list of dicts {box, score, label}
        """
        # validate image
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be an RGB array of shape (H,W,3)")

        # run model and get raw outputs
        inputs = self.processor(text=[queries], images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        h, w = image.shape[:2]
        target_sizes = torch.tensor([(h, w)]).to(self.device)
        # Use grounded post-process if available (new API), else fallback
        if hasattr(self.processor, "post_process_grounded_object_detection"):
            results = self.processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes)
        else:
            results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)
        res = results[0]
        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        labels = res["labels"].cpu().numpy()

        # Clip boxes to image bounds and compute areas
        boxes_clipped = boxes.copy()
        boxes_clipped[:, 0] = np.clip(boxes_clipped[:, 0], 0, w)
        boxes_clipped[:, 2] = np.clip(boxes_clipped[:, 2], 0, w)
        boxes_clipped[:, 1] = np.clip(boxes_clipped[:, 1], 0, h)
        boxes_clipped[:, 3] = np.clip(boxes_clipped[:, 3], 0, h)

        widths = (boxes_clipped[:, 2] - boxes_clipped[:, 0]).clip(min=0)
        heights = (boxes_clipped[:, 3] - boxes_clipped[:, 1]).clip(min=0)
        areas = widths * heights
        img_area = float(h * w)
        rel_areas = areas / (img_area + 1e-12)

        # initial keep by min area
        keep_mask = rel_areas >= min_box_rel_area
        if max_box_rel_area is not None:
            keep_mask = keep_mask & (rel_areas <= max_box_rel_area)

        # if a positive score_threshold is provided, apply it after potential auto-select (see below)
        if score_threshold > 0:
            keep_mask = keep_mask & (scores >= score_threshold)

        boxes = boxes_clipped[keep_mask]
        scores = scores[keep_mask]
        labels = labels[keep_mask]

        # if auto_select is requested, compute a dynamic threshold using the original raw scores
        if auto_select:
            # work with the raw (pre-filter) scores for threshold computation
            raw_scores = res["scores"].cpu().numpy()
            if raw_scores.size > 0:
                sorted_idx = np.argsort(-raw_scores)
                sorted_scores = raw_scores[sorted_idx]
                if auto_mode == "ratio":
                    top_score = float(sorted_scores[0])
                    dyn_thresh = max(top_score * float(auto_ratio), 0.0)
                else:
                    # knee: find largest relative drop between consecutive sorted scores
                    drops = sorted_scores[:-1] - sorted_scores[1:]
                    if drops.size == 0:
                        dyn_thresh = float(sorted_scores[0]) * float(auto_ratio)
                    else:
                        knee_idx = int(np.argmax(drops))
                        # threshold is the score after the largest drop
                        dyn_thresh = float(sorted_scores[knee_idx + 1])
                # ensure we keep at least min_keep candidates
                dyn_keep = (raw_scores >= dyn_thresh)
                if dyn_keep.sum() < min_keep:
                    # relax threshold to keep top min_keep
                    dyn_thresh = float(sorted_scores[min(min_keep, sorted_scores.size) - 1])

                # now re-apply keep mask using dyn_thresh but still respecting min_box_rel_area
                keep_mask = (res["labels"].cpu().numpy() >= 0) & (res["scores"].cpu().numpy() >= dyn_thresh)
                # filter by area as well
                keep_mask = keep_mask & (rel_areas >= min_box_rel_area)
                if max_box_rel_area is not None:
                    keep_mask = keep_mask & (rel_areas <= max_box_rel_area)
                boxes = boxes_clipped[keep_mask]
                scores = res["scores"].cpu().numpy()[keep_mask]
                labels = res["labels"].cpu().numpy()[keep_mask]

        # If requested, run per-label NMS (labels are indices into queries)
        if apply_nms and len(scores) > 0:
            keep_indices = []
            unique_labels = np.unique(labels)
            for ul in unique_labels:
                inds = np.where(labels == ul)[0]
                if inds.size == 0:
                    continue
                b = boxes[inds]
                s = scores[inds]
                order = np.argsort(-s)
                keep = self._nms_xyxy(b[order], s[order], iou_threshold)
                keep_indices.extend(inds[order[keep]].tolist())
            if keep_indices:
                sel = np.array(keep_indices, dtype=int)
                boxes = boxes[sel]
                scores = scores[sel]
                labels = labels[sel]
            else:
                boxes = np.zeros((0, 4))
                scores = np.zeros((0,))
                labels = np.zeros((0,), dtype=int)

        # Optionally keep only the best per query label
        if per_query_best and len(scores) > 0:
            keep_inds = []
            for ul in np.unique(labels):
                inds = np.where(labels == ul)[0]
                best = inds[np.argmax(scores[inds])]
                keep_inds.append(best)
            sel = np.array(keep_inds, dtype=int)
            boxes = boxes[sel]
            scores = scores[sel]
            labels = labels[sel]

        # final top-k by score
        if len(scores) > topk:
            idx = np.argsort(-scores)[:topk]
            boxes, scores, labels = boxes[idx], scores[idx], labels[idx]

        # format predictions
        predictions: List[Dict] = []
        for box, score, label_idx in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [float(v) for v in box]
            label = queries[int(label_idx)] if 0 <= int(label_idx) < len(queries) else str(int(label_idx))
            predictions.append({
                "box": (x1, y1, x2, y2),
                "score": float(score),
                "label": label,
            })

        return predictions

    @staticmethod
    def crop_from_box(image: np.ndarray, box: Tuple[float, float, float, float], margin_ratio: float = 0.0) -> np.ndarray:
        """Crop around box with optional margin (fraction of max(box_w, box_h))."""
        x1, y1, x2, y2 = [float(v) for v in box]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        m = max(0.0, float(margin_ratio)) * max(bw, bh)
        x1 -= m; y1 -= m; x2 += m; y2 += m
        h, w = image.shape[:2]
        x1 = int(max(0, np.floor(x1)))
        y1 = int(max(0, np.floor(y1)))
        x2 = int(min(w, np.ceil(x2)))
        y2 = int(min(h, np.ceil(y2)))
        if x2 <= x1: x2 = min(w, x1 + 1)
        if y2 <= y1: y2 = min(h, y1 + 1)
        return image[y1:y2, x1:x2]

    @staticmethod
    def _iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute IoU between a single box and an array of boxes. Boxes are (N,4) xyxy."""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter_w = np.maximum(0.0, x2 - x1)
        inter_h = np.maximum(0.0, y2 - y1)
        inter = inter_w * inter_h
        area_a = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
        area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_a + area_b - inter
        iou = np.zeros_like(inter)
        valid = union > 0
        iou[valid] = inter[valid] / union[valid]
        return iou

    def _nms_xyxy(self, boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
        """Greedy NMS. boxes: (N,4), scores: (N,). Returns list of kept indices relative to input order."""
        if boxes.size == 0:
            return []
        order = np.argsort(-scores)
        keep = []
        suppressed = np.zeros(len(order), dtype=bool)
        for i_idx, i in enumerate(order):
            if suppressed[i_idx]:
                continue
            keep.append(i_idx)
            if i_idx + 1 >= len(order):
                break
            i_box = boxes[i]
            rest_idx = np.arange(i_idx + 1, len(order))
            rest_boxes = boxes[order[rest_idx]]
            ious = self._iou_xyxy(i_box, rest_boxes)
            suppressed[rest_idx[ious > iou_thresh]] = True
        return keep

    @torch.no_grad()
    def get_raw_outputs(self, image: np.ndarray, queries: List[str]):
        """Run the model and return raw boxes/scores/labels (no post-filtering).

        Returns dict with numpy arrays: boxes (N,4), scores (N,), labels (N,)
        """
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be an RGB array of shape (H,W,3)")
        inputs = self.processor(text=[queries], images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        h, w = image.shape[:2]
        target_sizes = torch.tensor([(h, w)]).to(self.device)
        if hasattr(self.processor, "post_process_grounded_object_detection"):
            results = self.processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes)
        else:
            results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)
        res = results[0]
        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        labels = res["labels"].cpu().numpy()
        return {"boxes": boxes, "scores": scores, "labels": labels}

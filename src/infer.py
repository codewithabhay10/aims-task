import argparse
from typing import List
from pathlib import Path
import cv2
import numpy as np

from model import TextGrounder
from utils import draw_predictions, box_to_int


def main():
    parser = argparse.ArgumentParser(description="Zero-shot text grounding with OWLv2")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("query", type=str, nargs="+", help="Text query (use quotes); you can pass multiple")
    parser.add_argument("--threshold", type=float, default=0.25, help="Score threshold")
    parser.add_argument("--topk", type=int, default=5, help="Top-k detections to keep")
    parser.add_argument("--crop_out", type=str, default=None, help="Path to save best crop")
    parser.add_argument("--vis_out", type=str, default=None, help="Path to save visualization")

    args = parser.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    model = TextGrounder()

    preds = model.predict(image, args.query, score_threshold=args.threshold, topk=args.topk)
    if not preds:
        print("No matches above threshold.")
        return

    # Choose best by score
    best = max(preds, key=lambda p: p["score"])

    if args.crop_out:
        crop = model.crop_from_box(image, best["box"])
        Path(args.crop_out).parent.mkdir(parents=True, exist_ok=True)
        # save as BGR for OpenCV
        cv2.imwrite(args.crop_out, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        print(f"Saved crop -> {args.crop_out} {box_to_int(best['box'])} score={best['score']:.3f}")

    if args.vis_out:
        vis = draw_predictions(image, preds)
        Path(args.vis_out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.vis_out, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"Saved vis -> {args.vis_out}")

    # Always print results
    print("Detections:")
    for p in preds:
        print(f"  {p['label']}: box={box_to_int(p['box'])} score={p['score']:.3f}")


if __name__ == "__main__":
    main()

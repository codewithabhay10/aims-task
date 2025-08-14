import argparse
from pathlib import Path
import cv2
import numpy as np
from typing import List

from model import TextGrounder
from utils import draw_predictions, box_to_int


def collect_images(paths: List[str]) -> List[Path]:
    result = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                result.extend(path.rglob(ext))
        elif path.is_file():
            result.append(path)
    # de-duplicate and sort
    return sorted(set(result))


def main():
    parser = argparse.ArgumentParser(description="Batch zero-shot grounding")
    parser.add_argument("inputs", nargs="+", help="Image files and/or folders")
    parser.add_argument("--queries", required=True, nargs="+", help="One or more text queries (quote each)")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--out", type=str, default="outputs", help="Output root folder")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    model = TextGrounder()
    images = collect_images(args.inputs)
    if not images:
        print("No images found.")
        return

    print(f"Found {len(images)} images. Running with queries: {args.queries}")
    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[SKIP] Could not read: {img_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        preds = model.predict(rgb, args.queries, score_threshold=args.threshold, topk=args.topk)

        stem = img_path.stem
        vis = draw_predictions(rgb, preds)
        crop = None
        if preds:
            best = max(preds, key=lambda p: p["score"]) 
            crop = model.crop_from_box(rgb, best["box"]) 

        vis_out = out_root / f"{stem}_vis.jpg"
        cv2.imwrite(str(vis_out), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        if crop is not None:
            crop_out = out_root / f"{stem}_crop.jpg"
            cv2.imwrite(str(crop_out), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        print(f"[OK] {img_path.name} -> {vis_out.name}{' + crop' if crop is not None else ''}")


if __name__ == "__main__":
    main()

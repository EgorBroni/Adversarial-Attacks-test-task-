from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from imagenette_stage0.data import download_imagenette, make_loader
from imagenette_stage0.metrics import evaluate
from imagenette_stage0.models import make_model
from imagenette_stage0.utils import load_checkpoint, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a clean Stage 0 checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset-size", choices=["160", "320", "full"], default="160")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device="cpu")
    class_codes = list(ckpt["class_codes"])
    dataset_dir = download_imagenette(args.data_root, args.dataset_size)

    model = make_model(
        ckpt.get("arch", "convnext_tiny"),
        num_classes=len(class_codes),
        pretrained=False,
    )
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)

    loader = make_loader(
        dataset_dir,
        split="val",
        class_codes=class_codes,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=False,
    )
    clean = evaluate(
        model,
        loader,
        device,
        desc="validation clean",
        num_classes=len(class_codes),
    )
    result = {"checkpoint": args.checkpoint, "validation_clean": clean}

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_csv:
        path = Path(args.output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["subset", "metric", "accuracy", "total"],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerow(
                {
                    "subset": "training_classes",
                    "metric": "clean",
                    "accuracy": clean["accuracy"],
                    "total": clean["total"],
                }
            )


if __name__ == "__main__":
    main()

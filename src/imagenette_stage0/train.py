from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from torchvision.transforms import v2

from imagenette_stage0.data import (
    CLASS_NAMES,
    describe_codes,
    download_imagenette,
    make_loader,
    stage0_class_split,
)
from imagenette_stage0.metrics import evaluate
from imagenette_stage0.models import make_model
from imagenette_stage0.utils import (
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    set_seed,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a clean Imagenette Stage 0 model")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset-size", choices=["160", "320", "full"], default="160")
    parser.add_argument("--output-dir", default="runs/stage0_convnext_scratch")
    parser.add_argument("--training-class-count", type=int, default=8)
    parser.add_argument("--arch", choices=["resnet18", "convnext_tiny"], default="convnext_tiny")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--resume-from",
        default="",
        help="Load model weights from a Stage 0 checkpoint; optimizer and scheduler start fresh",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--trivial-augment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--random-erasing", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def checkpoint_payload(model, args, epoch: int, class_codes: list[str], metrics: dict) -> dict:
    return {
        "arch": args.arch,
        "epoch": epoch,
        "class_codes": class_codes,
        "class_names": {code: CLASS_NAMES[code] for code in class_codes},
        "state_dict": model.state_dict(),
        "metrics": metrics,
        "args": vars(args),
    }


def make_scheduler(optimizer, epochs: int, warmup_epochs: int):
    if not 0 <= warmup_epochs < epochs:
        raise ValueError("warmup_epochs must be non-negative and smaller than epochs")
    if warmup_epochs == 0:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def make_batch_augmentation(num_classes: int, mixup_alpha: float, cutmix_alpha: float):
    choices = []
    if mixup_alpha > 0:
        choices.append(v2.MixUp(alpha=mixup_alpha, num_classes=num_classes))
    if cutmix_alpha > 0:
        choices.append(v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes))
    if not choices:
        return None
    if len(choices) == 1:
        return choices[0]
    return v2.RandomChoice(choices)


def load_resume_weights(model, checkpoint_path: str, arch: str, class_codes: list[str]) -> dict:
    checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    checkpoint_arch = checkpoint.get("arch")
    checkpoint_classes = list(checkpoint.get("class_codes", []))
    if checkpoint_arch != arch:
        raise ValueError(
            f"Resume checkpoint architecture is {checkpoint_arch!r}, but --arch is {arch!r}"
        )
    if checkpoint_classes != class_codes:
        raise ValueError(
            "Resume checkpoint classes do not match the current Stage 0 class split: "
            f"{checkpoint_classes} != {class_codes}"
        )
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch: int,
    label_smoothing: float = 0.0,
    batch_augmentation=None,
) -> dict:
    model.train()
    total = 0
    correct = 0
    loss_sum = 0.0
    pbar = tqdm(loader, desc=f"train epoch {epoch + 1}", leave=False)

    for x, y in pbar:
        if batch_augmentation is not None:
            x, y = batch_augmentation(x, y)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)

        loss.backward()
        optimizer.step()

        metric_target = y.argmax(dim=1) if y.ndim == 2 else y
        batch = metric_target.numel()
        total += batch
        correct += int(logits.argmax(dim=1).eq(metric_target).sum().item())
        loss_sum += loss.item() * batch
        pbar.set_postfix(loss=loss_sum / max(total, 1), acc=correct / max(total, 1))

    return {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1)}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = download_imagenette(args.data_root, args.dataset_size)
    split = stage0_class_split(args.training_class_count)
    class_codes = split.training
    print(f"Device: {device}")
    print(f"Architecture: {args.arch}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Resume from: {args.resume_from or 'none'}")
    print(
        "Regularization: "
        f"label_smoothing={args.label_smoothing}, mixup_alpha={args.mixup_alpha}, "
        f"cutmix_alpha={args.cutmix_alpha}, trivial_augment={args.trivial_augment}, "
        f"random_erasing={args.random_erasing}"
    )
    print(f"Training classes: {describe_codes(class_codes)}")
    print(f"Held-out classes: {describe_codes(split.held_out)}")

    train_loader = make_loader(
        dataset_dir,
        split="train",
        class_codes=class_codes,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=True,
        trivial_augment=args.trivial_augment,
        random_erasing=args.random_erasing,
    )
    val_loader = make_loader(
        dataset_dir,
        split="val",
        class_codes=class_codes,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=False,
    )

    if args.resume_from and args.pretrained:
        raise ValueError("--resume-from and --pretrained cannot be used together")

    model = make_model(args.arch, num_classes=len(class_codes), pretrained=args.pretrained)
    if args.resume_from:
        resume_checkpoint = load_resume_weights(model, args.resume_from, args.arch, class_codes)
        print(
            f"Loaded model weights from epoch {resume_checkpoint.get('epoch', 'unknown')} "
            f"of {args.resume_from}"
        )
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)
    batch_augmentation = make_batch_augmentation(
        len(class_codes),
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
    )

    best_score = -1.0
    epochs_without_improvement = 0
    history_path = output_dir / "metrics.jsonl"
    history_path.unlink(missing_ok=True)
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            label_smoothing=args.label_smoothing,
            batch_augmentation=batch_augmentation,
        )
        clean_metrics = evaluate(
            model,
            val_loader,
            device,
            desc=f"val clean epoch {epoch + 1}",
            num_classes=len(class_codes),
        )
        record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val_clean": clean_metrics,
            "lr": scheduler.get_last_lr()[0],
        }
        scheduler.step()
        write_jsonl(history_path, record)
        print(json.dumps(record, indent=2, ensure_ascii=False))

        score = clean_metrics["accuracy"]
        payload = checkpoint_payload(model, args, epoch + 1, class_codes, record)
        save_checkpoint(output_dir / "last.pt", payload)
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "best.pt", payload)
        else:
            epochs_without_improvement += 1

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping after epoch {epoch + 1}: no validation improvement "
                f"for {epochs_without_improvement} epochs (best={best_score:.6f})"
            )
            break


if __name__ == "__main__":
    main()

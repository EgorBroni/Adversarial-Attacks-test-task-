from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "eval",
    num_classes: int | None = None,
) -> dict:
    model.eval()
    total = 0
    correct = 0
    if num_classes is None:
        num_classes = len(getattr(loader.dataset, "class_codes", []))
    class_total = torch.zeros(num_classes, dtype=torch.long)
    class_correct = torch.zeros(num_classes, dtype=torch.long)

    for x, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        batch_correct = pred.eq(y)
        total += y.numel()
        correct += int(batch_correct.sum().item())
        y_cpu = y.detach().cpu()
        correct_cpu = batch_correct.detach().cpu()
        class_total += torch.bincount(y_cpu, minlength=num_classes)
        class_correct += torch.bincount(y_cpu[correct_cpu], minlength=num_classes)

    per_class = {}
    codes = getattr(loader.dataset, "class_codes", [str(i) for i in range(num_classes)])
    for idx in range(num_classes):
        if class_total[idx].item() > 0:
            per_class[codes[idx]] = class_correct[idx].item() / class_total[idx].item()

    return {
        "accuracy": correct / max(total, 1),
        "total": total,
        "correct": correct,
        "per_class_accuracy": per_class,
    }

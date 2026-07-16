from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot learning curves from metrics.jsonl")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--title", default="Training history")
    parser.add_argument("--output", default="runs/training_curves.png")
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def load_records(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number} of {path}") from error
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def accuracy_series(record: dict) -> dict[str, float]:
    return {
        "train": record["train"]["accuracy"],
        "val_clean": record["val_clean"]["accuracy"],
    }


def tidy_history(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        epoch = record["epoch"]
        for name, value in accuracy_series(record).items():
            rows.append({"epoch": epoch, "metric": "accuracy", "series": name, "value": value})
        train_loss = record.get("train", {}).get("loss")
        if train_loss is not None:
            rows.append({"epoch": epoch, "metric": "loss", "series": "train", "value": train_loss})
        if record.get("lr") is not None:
            rows.append({"epoch": epoch, "metric": "learning_rate", "series": "lr", "value": record["lr"]})
    return pd.DataFrame(rows)


def plot_history(history: pd.DataFrame, title: str, output: str | Path) -> None:
    figure, (accuracy_ax, loss_ax) = plt.subplots(1, 2, figsize=(12, 4.5))

    accuracy = history[history["metric"] == "accuracy"]
    for name, values in accuracy.groupby("series"):
        accuracy_ax.plot(values["epoch"], values["value"], marker="o", label=name)
    accuracy_min = float(accuracy["value"].min())
    lower_bound = 0.0 if accuracy_min < 0.5 else max(0.0, accuracy_min - 0.05)
    accuracy_ax.set(
        title="Accuracy", xlabel="Epoch", ylabel="Top-1 accuracy", ylim=(lower_bound, 1)
    )
    accuracy_ax.grid(alpha=0.25)
    accuracy_ax.legend()

    loss = history[history["metric"] == "loss"]
    for name, values in loss.groupby("series"):
        loss_ax.plot(values["epoch"], values["value"], marker="o", label=name)
    loss_ax.set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy objective")
    loss_ax.grid(alpha=0.25)
    if not loss.empty:
        loss_ax.legend()

    figure.suptitle(title)
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output}")


def main() -> None:
    args = parse_args()
    history = tidy_history(load_records(args.jsonl))
    plot_history(history, args.title, args.output)
    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()

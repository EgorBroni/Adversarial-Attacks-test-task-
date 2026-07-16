import json

from imagenette_stage0.plot_training import load_records, tidy_history


def test_tidy_history_contains_stage0_curves(tmp_path):
    path = tmp_path / "metrics.jsonl"
    records = [
        {
            "epoch": 1,
            "train": {"loss": 1.2, "accuracy": 0.5},
            "val_clean": {"accuracy": 0.7},
            "lr": 0.001,
        },
        {
            "epoch": 2,
            "train": {"loss": 0.8, "accuracy": 0.6},
            "val_clean": {"accuracy": 0.75},
            "lr": 0.0005,
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    history = tidy_history(load_records(path))

    assert set(history["metric"]) == {"accuracy", "loss", "learning_rate"}
    assert "val_clean" in set(history["series"])
    assert "train" in set(history["series"])

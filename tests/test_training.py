import pytest
import torch

from imagenette_stage0.train import (
    load_resume_weights,
    make_batch_augmentation,
    make_scheduler,
)


def test_warmup_reaches_peak_before_cosine_decay():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = make_scheduler(optimizer, epochs=80, warmup_epochs=5)

    learning_rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates[0] == pytest.approx(1e-4)
    assert learning_rates[5] == pytest.approx(1e-3)
    assert learning_rates[6] < learning_rates[5]


def test_mixup_cutmix_produce_soft_class_targets():
    augmentation = make_batch_augmentation(num_classes=8, mixup_alpha=0.2, cutmix_alpha=1.0)
    images = torch.rand(8, 3, 32, 32)
    labels = torch.arange(8)

    augmented_images, soft_targets = augmentation(images, labels)

    assert augmented_images.shape == images.shape
    assert soft_targets.shape == (8, 8)
    assert torch.allclose(soft_targets.sum(dim=1), torch.ones(8))


def test_resume_loads_weights_for_matching_architecture_and_classes(tmp_path):
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "arch": "convnext_tiny",
            "class_codes": ["class_a", "class_b"],
            "state_dict": source.state_dict(),
            "epoch": 69,
        },
        checkpoint_path,
    )

    checkpoint = load_resume_weights(
        target,
        str(checkpoint_path),
        arch="convnext_tiny",
        class_codes=["class_a", "class_b"],
    )

    assert checkpoint["epoch"] == 69
    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)


def test_resume_rejects_different_class_split(tmp_path):
    model = torch.nn.Linear(3, 2)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "arch": "convnext_tiny",
            "class_codes": ["class_a", "class_b"],
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="classes do not match"):
        load_resume_weights(
            model,
            str(checkpoint_path),
            arch="convnext_tiny",
            class_codes=["class_a", "class_c"],
        )

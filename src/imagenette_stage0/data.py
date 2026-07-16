from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

CLASS_INFO: list[tuple[str, str]] = [
    ("n01440764", "tench"),
    ("n02102040", "English springer"),
    ("n02979186", "cassette player"),
    ("n03000684", "chain saw"),
    ("n03028079", "church"),
    ("n03394916", "French horn"),
    ("n03417042", "garbage truck"),
    ("n03425413", "gas pump"),
    ("n03445777", "golf ball"),
    ("n03888257", "parachute"),
]

CLASS_CODES = [code for code, _ in CLASS_INFO]
CLASS_NAMES = dict(CLASS_INFO)

IMAGENETTE_URLS = {
    "160": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
    "320": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
    "full": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
}


@dataclass(frozen=True)
class Stage0Split:
    training: list[str]
    held_out: list[str]


def stage0_class_split(training_class_count: int = 8) -> Stage0Split:
    if not 1 <= training_class_count < len(CLASS_CODES):
        raise ValueError("training_class_count must leave at least one held-out class")
    return Stage0Split(
        training=CLASS_CODES[:training_class_count],
        held_out=CLASS_CODES[training_class_count:],
    )


def describe_codes(codes: Sequence[str]) -> str:
    return ", ".join(f"{code}:{CLASS_NAMES.get(code, code)}" for code in codes)


def download_imagenette(data_root: str | Path, size: str = "160") -> Path:
    """Download and extract Imagenette if needed, returning the dataset directory."""

    if size not in IMAGENETTE_URLS:
        raise ValueError(f"Unknown Imagenette size {size!r}; expected one of {sorted(IMAGENETTE_URLS)}")

    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    dataset_name = "imagenette2" if size == "full" else f"imagenette2-{size}"
    dataset_dir = data_root / dataset_name
    if (dataset_dir / "train").exists() and (dataset_dir / "val").exists():
        return dataset_dir

    archive_path = data_root / f"{dataset_name}.tgz"
    if not archive_path.exists():
        print(f"Downloading {IMAGENETTE_URLS[size]} -> {archive_path}")
        urllib.request.urlretrieve(IMAGENETTE_URLS[size], archive_path)

    root_resolved = data_root.resolve()
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (data_root / member.name).resolve()
            if root_resolved not in member_path.parents and member_path != root_resolved:
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tar.extractall(data_root)
    return dataset_dir


def make_transforms(
    image_size: int,
    train: bool,
    trivial_augment: bool = False,
    random_erasing: float = 0.0,
):
    if train:
        operations = [
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        ]
        if trivial_augment:
            operations.append(transforms.TrivialAugmentWide())
        operations.append(transforms.ToTensor())
        if random_erasing > 0:
            operations.append(transforms.RandomErasing(p=random_erasing))
        return transforms.Compose(operations)
    resize_to = max(image_size + 32, int(image_size * 1.15))
    return transforms.Compose(
        [
            transforms.Resize(resize_to),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


class FilteredImageFolder(Dataset):
    """ImageFolder subset restricted to the Stage 0 training classes."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        class_codes: Sequence[str],
        transform=None,
    ) -> None:
        split_root = Path(root) / split
        if not split_root.exists():
            raise FileNotFoundError(f"Split directory not found: {split_root}")

        self.base = datasets.ImageFolder(split_root)
        self.loader = self.base.loader
        self.transform = transform
        self.class_codes = list(class_codes)
        self.class_to_idx = {code: idx for idx, code in enumerate(self.class_codes)}

        samples = []
        for path, _base_target in self.base.samples:
            code = Path(path).parent.name
            if code in self.class_to_idx:
                samples.append((path, self.class_to_idx[code]))

        self.samples = sorted(samples, key=lambda item: item[0])
        self.targets = [target for _, target in self.samples]
        if not self.samples:
            raise ValueError(f"No samples found for classes {self.class_codes} in {split_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def make_loader(
    dataset_dir: str | Path,
    split: str,
    class_codes: Sequence[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
    trivial_augment: bool = False,
    random_erasing: float = 0.0,
) -> DataLoader:
    dataset = FilteredImageFolder(
        dataset_dir,
        split=split,
        class_codes=class_codes,
        transform=make_transforms(
            image_size,
            train=train,
            trivial_augment=trivial_augment,
            random_erasing=random_erasing,
        ),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def class_counts(codes: Iterable[str], dataset_dir: str | Path, split: str) -> dict[str, int]:
    split_root = Path(dataset_dir) / split
    return {code: len(list((split_root / code).glob("*"))) for code in codes}

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights

from paths import resolve_cv_paths

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cct_ds_challenge.train_cv import RunConfig

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# ============================================================
# DATA LOADING
# ============================================================






def validate_dataset_structure(paths: Dict[str, Path]) -> None:
    """Validate that the expected yes/no folder structure exists."""
    image_dir = paths["image_dir"]
    yes_dir = image_dir / "yes"
    no_dir = image_dir / "no"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not yes_dir.exists():
        raise FileNotFoundError(f"Expected positive class directory: {yes_dir}")
    if not no_dir.exists():
        raise FileNotFoundError(f"Expected negative class directory: {no_dir}")


def list_valid_images(folder: Path) -> List[Path]:
    """List supported image files in a folder."""
    return sorted([path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES])


def load_labels_dataframe(cfg: RunConfig) -> pd.DataFrame:
    """Build the labelled image dataframe from yes/no folders."""
    paths = resolve_cv_paths(
    project_root=cfg.project_root,
    image_dir=cfg.image_dir,
    output_dir=cfg.output_dir,
    )
    validate_dataset_structure(paths)

    yes_dir = paths["image_dir"] / "yes"
    no_dir = paths["image_dir"] / "no"

    rows = []
    for folder, label in [(yes_dir, 1), (no_dir, 0)]:
        for image_path in list_valid_images(folder):
            rows.append(
                {
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "label": label,
                    "class_name": "yes" if label == 1 else "no",
                }
            )

    label_df = pd.DataFrame(rows)
    if label_df.empty:
        raise RuntimeError(f"No images found under {paths['image_dir']} with supported suffixes {sorted(ALLOWED_SUFFIXES)}")

    class_counts = label_df["label"].value_counts().to_dict()
    if len(class_counts) < 2:
        raise RuntimeError(f"Expected two classes. Found counts: {class_counts}")

    unreadable_files = []
    sample_paths = label_df["image_path"].sample(min(20, len(label_df)), random_state=cfg.seed).tolist()
    for sample_path in sample_paths:
        try:
            with Image.open(sample_path) as img:
                img.verify()
        except (UnidentifiedImageError, OSError) as exc:
            unreadable_files.append((sample_path, str(exc)))
    if unreadable_files:
        raise RuntimeError(f"Found unreadable images, e.g. {unreadable_files[:3]}")

    print(f"Loaded {len(label_df)} images from {paths['image_dir']}")
    print(f"Class counts: {label_df['label'].value_counts().sort_index().to_dict()}")
    return label_df


def make_balanced_subset(label_df: pd.DataFrame, sample_per_class: int, seed: int) -> pd.DataFrame:
    """Create a balanced development subset for faster smoke testing."""
    sampled_frames = []
    rng = np.random.default_rng(seed)
    for label, group_df in label_df.groupby("label"):
        sample_size = min(sample_per_class, len(group_df))
        sampled_indices = rng.choice(group_df.index.values, size=sample_size, replace=False)
        sampled_frames.append(group_df.loc[sampled_indices])
    subset_df = pd.concat(sampled_frames, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"Development subset class counts: {subset_df['label'].value_counts().to_dict()}")
    return subset_df


def stratified_train_val_test_split(
    label_df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a stratified train/validation/test split."""
    train_df, temp_df = train_test_split(
        label_df,
        test_size=(val_size + test_size),
        stratify=label_df["label"],
        random_state=seed,
    )
    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["label"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ============================================================
# TRANSFORMS AND DATALOADERS
# ============================================================


class PoolDataset(Dataset):
    """Dataset wrapper for labelled swimming-pool tiles."""

    def __init__(self, label_df: pd.DataFrame, transform=None):
        self.label_df = label_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.label_df)

    def __getitem__(self, idx: int):
        row = self.label_df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return image, label, row["image_path"]


def get_transforms(
    image_size: int,
    improved: bool = False,
    use_improved_augmentation: bool = True,
    improved_resize_padding: int = 16,
    improved_rotation_degrees: float = 10.0,
):
    """Build train/eval transforms for baseline or improved training."""
    weights = ResNet18_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    if not improved:
        train_tfms = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        if use_improved_augmentation:
            padded_size = image_size + improved_resize_padding
            train_tfms = transforms.Compose(
                [
                    transforms.Resize((padded_size, padded_size)),
                    transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.2),
                    transforms.RandomRotation(degrees=improved_rotation_degrees),
                    transforms.ColorJitter(
                        brightness=0.10,
                        contrast=0.10,
                        saturation=0.05,
                        hue=0.02,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            train_tfms = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tfms, eval_tfms


def make_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_size: int,
    batch_size: int,
    num_workers: int,
    improved: bool = False,
    use_improved_augmentation: bool = True,
    improved_resize_padding: int = 16,
    improved_rotation_degrees: float = 10.0,
):
    """Build train, validation, and test dataloaders."""
    train_transforms, eval_transforms = get_transforms(
        image_size=image_size,
        improved=improved,
        use_improved_augmentation=use_improved_augmentation,
        improved_resize_padding=improved_resize_padding,
        improved_rotation_degrees=improved_rotation_degrees)
    train_dataset = PoolDataset(train_df, transform=train_transforms)
    val_dataset = PoolDataset(val_df, transform=eval_transforms)
    test_dataset = PoolDataset(test_df, transform=eval_transforms)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


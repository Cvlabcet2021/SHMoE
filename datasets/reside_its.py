import os
from pathlib import Path

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class RESIDEITS(Dataset):
    """
    RESIDE Indoor Training Set (ITS)

    This dataset publically available
    """

    def __init__(
        self,
        hazy_dir,
        gt_dir,
        image_size=512,
        crop_size=512,
        train=True
    ):
        self.hazy_dir = Path(hazy_dir)
        self.gt_dir = Path(gt_dir)

        self.image_size = image_size
        self.crop_size = crop_size
        self.train = train

        self.extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff"
        }

        self.hazy_images = sorted(
            [
                p for p in self.hazy_dir.iterdir()
                if p.suffix.lower() in self.extensions
            ]
        )

        if len(self.hazy_images) == 0:
            raise RuntimeError(
                f"No hazy images found in: {self.hazy_dir}"
            )

        self.pairs = []

        for hazy_path in self.hazy_images:

            gt_path = self._find_gt_image(hazy_path)

            if gt_path is not None:
                self.pairs.append(
                    (hazy_path, gt_path)
                )

        if len(self.pairs) == 0:
            raise RuntimeError(
                "No matching hazy/ground-truth image pairs found."
            )

        print(
            f"Loaded {len(self.pairs)} image pairs "
            f"from {self.hazy_dir}"
        )

    # ---------------------------------------------------------
    # Find corresponding ground-truth image
    # ---------------------------------------------------------

    def _find_gt_image(self, hazy_path):

        stem = hazy_path.stem

        candidates = [
            self.gt_dir / f"{stem}.jpg",
            self.gt_dir / f"{stem}.jpeg",
            self.gt_dir / f"{stem}.png",
            self.gt_dir / f"{stem}.bmp",
            self.gt_dir / f"{stem}.tif",
            self.gt_dir / f"{stem}.tiff",
        ]

        for candidate in candidates:

            if candidate.exists():
                return candidate

        return None

    # ---------------------------------------------------------
    # Dataset length
    # ---------------------------------------------------------

    def __len__(self):

        return len(self.pairs)

    # ---------------------------------------------------------
    # Load image
    # ---------------------------------------------------------

    @staticmethod
    def _load_image(path):

        image = Image.open(path).convert("RGB")

        return image

    # ---------------------------------------------------------
    # Paired random crop
    # ---------------------------------------------------------

    def _random_crop(self, hazy, gt):

        width, height = hazy.size

        crop = self.crop_size

        # If image is smaller than crop size,
        # resize both images first.

        if width < crop or height < crop:

            new_width = max(width, crop)
            new_height = max(height, crop)

            hazy = hazy.resize(
                (new_width, new_height),
                Image.BICUBIC
            )

            gt = gt.resize(
                (new_width, new_height),
                Image.BICUBIC
            )

            width, height = hazy.size

        if width == crop and height == crop:
            return hazy, gt

        left = torch.randint(
            0,
            width - crop + 1,
            (1,)
        ).item()

        top = torch.randint(
            0,
            height - crop + 1,
            (1,)
        ).item()

        hazy = TF.crop(
            hazy,
            top,
            left,
            crop,
            crop
        )

        gt = TF.crop(
            gt,
            top,
            left,
            crop,
            crop
        )

        return hazy, gt

    # ---------------------------------------------------------
    # Paired augmentation
    # ---------------------------------------------------------

    def _augment(self, hazy, gt):

        # Horizontal flip
        if torch.rand(1).item() > 0.5:

            hazy = TF.hflip(hazy)
            gt = TF.hflip(gt)

        # Vertical flip
        if torch.rand(1).item() > 0.5:

            hazy = TF.vflip(hazy)
            gt = TF.vflip(gt)

        # 90-degree rotation
        k = torch.randint(
            0,
            4,
            (1,)
        ).item()

        if k > 0:

            angle = 90 * k

            hazy = TF.rotate(
                hazy,
                angle
            )

            gt = TF.rotate(
                gt,
                angle
            )

        return hazy, gt

    # ---------------------------------------------------------
    # Validation/test resizing
    # ---------------------------------------------------------

    def _resize(self, hazy, gt):

        hazy = hazy.resize(
            (self.image_size, self.image_size),
            Image.BICUBIC
        )

        gt = gt.resize(
            (self.image_size, self.image_size),
            Image.BICUBIC
        )

        return hazy, gt

    # ---------------------------------------------------------
    # Convert to tensor
    # ---------------------------------------------------------

    @staticmethod
    def _to_tensor(image):

        return TF.to_tensor(image)

    # ---------------------------------------------------------
    # Get item
    # ---------------------------------------------------------

    def __getitem__(self, index):

        hazy_path, gt_path = self.pairs[index]

        hazy = self._load_image(hazy_path)
        gt = self._load_image(gt_path)

        if self.train:

            hazy, gt = self._random_crop(
                hazy,
                gt
            )

            hazy, gt = self._augment(
                hazy,
                gt
            )

        else:

            hazy, gt = self._resize(
                hazy,
                gt
            )

        hazy = self._to_tensor(hazy)
        gt = self._to_tensor(gt)

        return {
            "hazy": hazy,
            "gt": gt,
            "name": hazy_path.name
        }


# =============================================================
# Dataset builder
# =============================================================

def build_reside_dataset(
    hazy_dir,
    gt_dir,
    image_size=512,
    crop_size=512,
    train=True
):

    dataset = RESIDEITS(
        hazy_dir=hazy_dir,
        gt_dir=gt_dir,
        image_size=image_size,
        crop_size=crop_size,
        train=train
    )

    return dataset


# =============================================================
# Simple test
# =============================================================

if __name__ == "__main__":

    dataset = RESIDEITS(
        hazy_dir="./data/RESIDE-ITS/train/hazy",
        gt_dir="./data/RESIDE-ITS/train/gt",
        image_size=512,
        crop_size=512,
        train=True
    )

    sample = dataset[0]

    print("Hazy:", sample["hazy"].shape)
    print("GT:", sample["gt"].shape)
    print("Name:", sample["name"])

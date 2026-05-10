"""Grab a single CIFAR-10 sample and save it as tests/test_image.jpg."""
from __future__ import annotations

import argparse
from pathlib import Path

from torchvision import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tests/test_image.jpg")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    ds = datasets.CIFAR10(args.data_dir, train=False, download=True)
    img, label = ds[args.index]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, format="JPEG", quality=95)
    print(f"saved {out} (label={ds.classes[label]})")


if __name__ == "__main__":
    main()

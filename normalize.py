"""Calculate the mean and standard deviation of the given dataset.

Based on https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html

Usage:
    python normalize.py --data_dir /path/to/dataset --workers N

Notes:
    - The dataset must be in Torchvision's ImageFolder format.
    - The mean and standard deviation are computed over the entire dataset. Images are
      resized so that the shortest edge is 256 px and then center-cropped to
      224 x 224 px before the mean and standard deviation are computed.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(prog="normalize.py")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="The dataset directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="The number of workers to use for data loading",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create a DataLoader for the dataset
    transform = transforms.Compose(
        [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    dataset = datasets.ImageFolder(os.path.join(args.data_dir), transform=transform)
    data_loader = DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=args.workers
    )

    # Placeholders
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # Loop through images
    for batch in data_loader:
        inputs, _ = batch
        psum += inputs.sum(axis=[0, 2, 3])
        psum_sq += (inputs**2).sum(axis=[0, 2, 3])

    # Pixel count
    count = len(dataset) * 224 * 224

    # Mean and std. dev.
    mean = psum / count
    var = (psum_sq / count) - (mean**2)
    std = torch.sqrt(var)

    # output
    print("Mean: " + str(mean))
    print("Std. Dev.:  " + str(std))


if __name__ == "__main__":
    main()

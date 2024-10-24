#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SliceDataset
from utils import class2one_hot
from operator import itemgetter

def count_pixels_per_class(dataloader, K):
    """
    Count the total number of pixels for each class in the dataset.
    """
    class_counts = np.zeros(K)
    
    for batch in tqdm(dataloader, desc="Processing batches"):
        gts = torch.argmax(batch['gts'], dim=1)  # Ground truth masks
        for k in range(K):
            class_counts[k] += (gts == k).sum().item()
    
    return class_counts

def plot_class_counts(class_counts, class_names, output_file):
    """
    Plot a bar chart of background pixels vs total of all other classes.
    """
    background_count = class_counts[0]
    other_classes_count = class_counts[1:].sum()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(['Background', 'Other Classes'], [background_count, other_classes_count], color='skyblue', width=0.2)
    
    # Add labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center')  # va: vertical alignment, ha: horizontal alignment
    
    plt.ylabel('Pixel Count')
    plt.title('Background Pixels vs Total of All Other Classes')
    plt.savefig(output_file)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=Path, required=True, help="Output file to save the bar chart.")
    args = parser.parse_args()

    # Parameters for SEGTHOR dataset
    K = 5
    class_names = ["background", "esophagus", "heart", "trachea", "aorta"]

    root_dir = Path("data") / "SEGTHORCORRECT"
    target_size = (256, 256)
    batch_size = 64

    img_transform = transforms.Compose([
        transforms.Resize(target_size),  # Resize the image to the target size
        lambda img: img.convert('L'), # Convert to grayscale
        lambda img: np.array(img)[np.newaxis, ...], # Add one dimension to simulate batch
        lambda nd: nd / 255,  # max <= 1 # Normalize the image
        lambda nd: torch.tensor(nd, dtype=torch.float32) # Convert to tensor
    ])

    gt_transform = transforms.Compose([
        transforms.Resize(target_size, interpolation=InterpolationMode.NEAREST),
        lambda img: np.array(img)[...],
        lambda nd: nd / 63,  # Normalize the image for SEGTHOR
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],
        lambda t: class2one_hot(t, K=K),
        itemgetter(0)
    ])

    dataset = SliceDataset('train', root_dir, img_transform=img_transform, gt_transform=gt_transform, debug=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    class_counts = count_pixels_per_class(dataloader, K)
    plot_class_counts(class_counts, class_names, args.output_file)

if __name__ == '__main__':
    main()
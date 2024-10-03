import numpy as np
from skimage.transform import rotate, rescale, AffineTransform, warp
from skimage import exposure
from scipy.ndimage import gaussian_filter, shift
from skimage.io import imsave, imread
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from pathlib import Path
import random
import argparse
from skimage.exposure import adjust_gamma
from skimage.exposure import rescale_intensity
from tqdm import tqdm

def add_gaussian_noise(image, mean=0, std=0.05):
    """
    Adds Gaussian noise to the image.
    """
    noisy_img = image + np.random.normal(mean, std * 255, image.shape).astype(np.uint8)
    return np.clip(noisy_img, 0, 255)  # Ensure pixel values remain in valid range [0, 255]

def apply_gaussian_blur(image, sigma=1):
    """
    Applies Gaussian blur to the image.
    """
    blurred_img = gaussian_filter(image, sigma=sigma)
    return blurred_img.astype(np.uint8)

def adjust_brightness(image, gamma=1.0):
    """
    Adjusts the brightness of the image using gamma correction.
    gamma < 1: makes the image brighter
    gamma > 1: makes the image darker
    """
    return adjust_gamma(image, gamma).astype(np.uint8)

def adjust_contrast(image, low=2, high=98):
    """
    Adjusts the contrast of the image by rescaling intensity.
    `low` and `high` are percentiles for contrast stretching.
    """
    p2, p98 = np.percentile(image, (low, high))
    return rescale_intensity(image, in_range=(p2, p98)).astype(np.uint8)

def augment_noise_contrast(image):
    """
    Apply Gaussian noise and contrast adjustment.
    """
    # Gaussian Noise
    image = add_gaussian_noise(image, mean=0, std=0.05)

    # Contrast Adjustment
    image = adjust_contrast(image, low=2, high=98)

    return image

def augment_blur_brightness(image):
    """
    Apply Gaussian blur and brightness adjustment.
    """
    # Gaussian Blur
    image = apply_gaussian_blur(image, sigma=random.uniform(0.5, 1.5))

    # Brightness Adjustment
    image = adjust_brightness(image, gamma=random.uniform(0.7, 1.3))

    return image

def augment_and_save(image_dir, gt_dir, aug_image_dir, aug_gt_dir, num_augmentations=3):
    """
    Augment the dataset with 3 noise+contrast and 3 blur+brightness transformations and save.
    """
    # Ensure the output directories exist
    aug_image_dir.mkdir(parents=True, exist_ok=True)
    aug_gt_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(image_dir.glob("*.png")))
    gt_paths = sorted(list(gt_dir.glob("*.png")))

    for img_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths), desc="Augmenting Intensity"):
        img = np.array(imread(img_path))
        gt = np.array(imread(gt_path))  # Ground truth doesn't need intensity transformations

        # Apply 3 noise + contrast augmentations
        for i in range(num_augmentations):
            aug_img = augment_noise_contrast(img)
            img_filename = img_path.stem + f"_noise_contrast_aug{i:03d}.png"
            gt_filename = gt_path.stem + f"_noise_contrast_aug{i:03d}.png"
            imsave(str(aug_image_dir / img_filename), aug_img)
            imsave(str(aug_gt_dir / gt_filename), gt)

        # Apply 3 blur + brightness augmentations
        for i in range(num_augmentations):
            aug_img = augment_blur_brightness(img)
            img_filename = img_path.stem + f"_blur_brightness_aug{i:03d}.png"
            gt_filename = gt_path.stem + f"_blur_brightness_aug{i:03d}.png"
            imsave(str(aug_image_dir / img_filename), aug_img)
            imsave(str(aug_gt_dir / gt_filename), gt)


def main():
    parser = argparse.ArgumentParser(description='Augment dataset with realistic transformations.')
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory containing the SEGTHOR dataset')
    parser.add_argument('--num_augmentations', type=int, default=3, help='Number of augmentations per image')

    args = parser.parse_args()

    # Define the data directories
    data_dir = Path(args.data_dir)
    train_img_dir = data_dir / "train" / "img_preprocessed"
    train_gt_dir = data_dir / "train" / "gt_preprocessed"

    # Define the augmented directories
    aug_train_img_dir = data_dir / "train" / "img_pre_intensity_aug"
    aug_train_gt_dir = data_dir / "train" / "gt_pre_intensity_aug"

    # Augment and save the training set
    print("Augmenting training set...")
    augment_and_save(train_img_dir, train_gt_dir, aug_train_img_dir, aug_train_gt_dir, args.num_augmentations)

if __name__ == "__main__":
    main()
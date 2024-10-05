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
from tqdm import tqdm

# Function for elastic deformation
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    alpha: Control the intensity of the deformation
    sigma: Standard deviation for the Gaussian filter
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def augment_image(img, gt, rotation_angle=5, scale_factor=1.05, translation=(5, 5), elastic=True):
    """
    Apply augmentations: rotation, scaling, translation, and elastic deformation (optional)
    """
    # Rotation (small rotation around the center)
    img_rot = rotate(img, rotation_angle, resize=False, mode='edge', preserve_range=True).astype(np.uint8)
    gt_rot = rotate(gt, rotation_angle, resize=False, mode='edge', order=0, preserve_range=True).astype(np.uint8)

    # Scaling/Zooming (slight zooming or shrinking)
    img_scaled = rescale(img_rot, scale_factor, mode='constant', anti_aliasing=False, preserve_range=True).astype(np.uint8)
    gt_scaled = rescale(gt_rot, scale_factor, mode='constant', order=0, preserve_range=True).astype(np.uint8)

    # Translation (minor shifts)
    img_translated = shift(img_scaled, translation, mode='nearest').astype(np.uint8)
    gt_translated = shift(gt_scaled, translation, mode='nearest').astype(np.uint8)

    # Optional elastic deformation (simulating breathing or heart movements)
    if elastic:
        img_elastic = elastic_transform(img_translated, alpha=20, sigma=4)
        gt_elastic = elastic_transform(gt_translated, alpha=20, sigma=4)
    else:
        img_elastic, gt_elastic = img_translated, gt_translated

    return img_elastic, gt_elastic

def augment_and_save(image_dir, gt_dir, aug_image_dir, aug_gt_dir, num_augmentations=5, elastic=True):
    """
    Augment the dataset and save the augmented images in the same structure.
    """
    aug_image_dir.mkdir(parents=True, exist_ok=True)
    aug_gt_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(image_dir.glob("*.png")))
    gt_paths = sorted(list(gt_dir.glob("*.png")))

    for img_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths), desc="Augmenting data"):
        # Load the images and ground truth
        img = np.array(imread(img_path))
        gt = np.array(imread(gt_path))

        for i in range(num_augmentations):
            # Randomize augmentation parameters
            rotation_angle = random.uniform(-5, 5)  # Small rotation in degrees
            scale_factor = random.uniform(0.95, 1.05)  # Slight zoom in/out
            translation = [random.uniform(-5, 5), random.uniform(-5, 5)]  # Small translations

            # Apply augmentations (with or without elastic deformation)
            aug_img, aug_gt = augment_image(img, gt, rotation_angle, scale_factor, translation, elastic=elastic)

            # Save augmented images
            img_filename = img_path.stem + f"_aug{i:03d}.png"
            gt_filename = gt_path.stem + f"_aug{i:03d}.png"
            imsave(str(aug_image_dir / img_filename), aug_img)
            imsave(str(aug_gt_dir / gt_filename), aug_gt)

def main():
    parser = argparse.ArgumentParser(description='Augment dataset with realistic transformations.')
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory containing the SEGTHOR dataset')
    parser.add_argument('--num_augmentations', type=int, default=5, help='Number of augmentations per image')
    parser.add_argument('--elastic', action='store_true', help='Apply elastic deformation')

    args = parser.parse_args()

    # Define the data directories
    data_dir = Path(args.data_dir)
    train_img_dir = data_dir / "train" / "img_preprocessed"
    train_gt_dir = data_dir / "train" / "gt_preprocessed"

    # Define the augmented directories
    aug_train_img_dir = data_dir / "train" / "img_pre_spatial_aug"
    aug_train_gt_dir = data_dir / "train" / "gt_pre_spatial_aug"

    # Augment and save the training set
    print("Augmenting training set...")
    augment_and_save(train_img_dir, train_gt_dir, aug_train_img_dir, aug_train_gt_dir, args.num_augmentations, elastic=args.elastic)


if __name__ == "__main__":
    main()
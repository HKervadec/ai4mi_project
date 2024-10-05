from skimage import exposure
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import os
from PIL import Image
from skimage import measure
from skimage.morphology import remove_small_objects

VALID_LABELS = {0, 1, 2, 3, 4} # background esophagus heart trachea aorta

def resize_image_and_label(image, label, target_size=(256, 256)):
    """
    Resize image and segmentation label to a fixed target size.
    Use nearest-neighbor interpolation for the label to preserve discrete class values.
    
    Parameters:
    - image: The original image (numpy array).
    - label: The ground truth mask (numpy array).
    - target_size: The desired size of the output image and mask (default: 256x256).
    
    Returns:
    - resized_image: The resized image.
    - resized_label: The resized segmentation label.
    """
    resized_image = resize(image, target_size, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    resized_label = resize(label, target_size, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)  # Nearest-neighbor interpolation for labels
    return resized_image, resized_label

def crop_and_resize(image, mask, padding=10, target_size=256, body_threshold=7, min_size=500):
    """
    Crop the image and mask to a square bounding box around the body cavity,
    excluding small, isolated regions (e.g., lens or noise).
    
    Parameters:
    - image: The original image (numpy array).
    - mask: The ground truth mask (numpy array).
    - padding: The number of pixels to expand the bounding box by (default: 10).
    - target_size: The desired size of the square cropped output (default: 256x256).
    - body_threshold: Pixel intensity threshold to detect body cavity.
    - min_size: Minimum size for connected components to be kept (default: 500 pixels).
    
    Returns:
    - cropped_resized_image: The cropped and resized image.
    - cropped_resized_mask: The cropped and resized ground truth mask.
    """
    # Step 1: Create a mask for the body cavity (non-background pixels in the image)
    body_mask = image > body_threshold  # Threshold to detect the body cavity

    # Step 2: Remove small objects (isolated regions)
    body_mask_cleaned = remove_small_objects(body_mask, min_size=min_size)

    # Step 3: Find the coordinates of the bounding box around the largest connected component
    labels = measure.label(body_mask_cleaned)  # Label connected regions
    largest_component = (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1)  # Find the largest component

    coords = np.argwhere(largest_component)
    
    if coords.size == 0:
        # If no body cavity is detected, return the original image and mask
        return image, mask

    # Step 4: Get the min and max coordinates for the bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Step 5: Add padding to the bounding box
    y_min = max(0, y_min - padding)
    x_min = max(0, x_min - padding)
    y_max = min(image.shape[0], y_max + padding)
    x_max = min(image.shape[1], x_max + padding)

    # Step 6: Ensure the crop is square by adjusting either width or height
    width = x_max - x_min
    height = y_max - y_min
    side_length = max(width, height)  # We want a square, so take the larger of the two

    # Adjust the x (width) dimension if needed
    if width < side_length:
        diff = side_length - width
        x_min = max(0, x_min - diff // 2)
        x_max = min(image.shape[1], x_max + diff // 2)

    # Adjust the y (height) dimension if needed
    if height < side_length:
        diff = side_length - height
        y_min = max(0, y_min - diff // 2)
        y_max = min(image.shape[0], y_max + diff // 2)

    # Step 7: Crop the image and the mask to the square region
    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    # Step 8: Resize the cropped square to the target size (e.g., 256x256)
    cropped_resized_image = np.array(Image.fromarray(cropped_image).resize((target_size, target_size), Image.BILINEAR))
    cropped_resized_mask = np.array(Image.fromarray(cropped_mask).resize((target_size, target_size), Image.NEAREST))

    return cropped_resized_image, cropped_resized_mask

# Intensity normalization (rescaling pixel values to [0, 1])
def normalize_intensity(image):
    """
    Normalize image intensity values to the range [0, 1].
    """
    return (image - np.min(image)) / (np.max(image) - np.min(image))

# Histogram equalization using CLAHE (for better contrast)
def apply_clahe(image, clip_limit=0.03):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance contrast.
    """
    return exposure.equalize_adapthist(image, clip_limit=clip_limit)

# Optional: Smoothing/Filtering (Gaussian smoothing for noise reduction)
def smooth_image(image, sigma=1):
    return gaussian_filter(image, sigma=sigma)


# Combined Pre-processing function for images and labels
def preprocess_image_and_label(image, label, padding=10, target_size=256, body_threshold=7, min_size=500, crop=False):
    """
    Apply the full pre-processing pipeline:
    1. Crop and resize image and label.
    2. Normalize intensity (for images only, not labels).
    3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    4. Optionally, apply Gaussian smoothing.

    The label should not undergo intensity transformations.
    """
    # Step 1: Crop and resize image and label, or just resize based on the 'crop' flag
    if crop:
        image_cropped, label_cropped = crop_and_resize(image, label, padding=padding, target_size=target_size, body_threshold=body_threshold, min_size=min_size)
    else:
        image_cropped, label_cropped = resize_image_and_label(image, label, target_size=(target_size, target_size))


    # Step 2: Normalize intensity (for images only, not labels)
    image_normalized = normalize_intensity(image_cropped)

    # Step 3: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
    image_clahe = apply_clahe(image_normalized)

    return image_clahe, label_cropped


# Pre-process and save the dataset
def preprocess_and_save(image_dir, label_dir, output_img_dir, output_label_dir, padding=10, target_size=256, body_threshold=7, min_size=500, crop=False):
    """
    Pre-process all images and segmentation labels (ground truths), and save them to the specified output directories.
    """
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(image_dir.glob("*.png")))
    label_paths = sorted(list(label_dir.glob("*.png")))

    for img_path, label_path in tqdm(zip(image_paths, label_paths), total=len(image_paths), desc="Pre-processing dataset"):
        img = np.array(imread(img_path))
        label = np.array(imread(label_path))

        # Apply the full pre-processing pipeline
        img_preprocessed, label_preprocessed = preprocess_image_and_label(img, label, padding=padding, target_size=target_size, body_threshold=body_threshold, min_size=min_size, crop=crop)

        # Save the pre-processed image and label
        img_filename = output_img_dir / img_path.name
        label_filename = output_label_dir / label_path.name

        imsave(str(img_filename), (img_preprocessed * 255).astype(np.uint8))  # Rescale intensity back to [0, 255] for saving
        imsave(str(label_filename), label_preprocessed)


def main():
    parser = argparse.ArgumentParser(description='Pre-process the SegTHOR dataset with cropping and intensity adjustments.')
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory containing the SEGTHOR dataset')
    parser.add_argument('--padding', type=int, default=10, help='Padding around the body cavity during cropping')
    parser.add_argument('--target_size', type=int, default=256, help='Target size for resized images (default: 256)')
    parser.add_argument('--body_threshold', type=int, default=7, help='Padding around the body cavity during cropping')
    parser.add_argument('--min_size', type=int, default=500, help='Padding around the body cavity during cropping')
    parser.add_argument('--crop', action='store_true', help='Use crop and resize, otherwise just resize')

    args = parser.parse_args()

    # Define the data directories
    data_dir = Path(args.data_dir)
    train_img_dir = data_dir / "train" / "img"
    train_gt_dir = data_dir / "train" / "gt"

    # Define the output directories for the preprocessed data
    preprocessed_train_img_dir = data_dir / "train" / "img_preprocessed"
    preprocessed_train_gt_dir = data_dir / "train" / "gt_preprocessed"

    # Pre-process and save the training set
    print("Pre-processing training set...")
    preprocess_and_save(train_img_dir, train_gt_dir, preprocessed_train_img_dir, preprocessed_train_gt_dir, padding=args.padding, target_size=args.target_size, body_threshold=args.body_threshold, min_size=args.min_size, crop=args.crop)

if __name__ == "__main__":
    main()
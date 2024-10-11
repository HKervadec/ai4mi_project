import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import dilation, erosion, disk, ball
import nibabel as nib
import argparse
import torch
from pathlib import Path
from scipy.ndimage import label
from scipy.ndimage import gaussian_filter


def keep_largest_components(volume):
    refined_volume = np.zeros_like(volume)
    for label_value in np.unique(volume):
        if label_value == 0:
            continue  # Skip background
        # Binary mask for the current label
        mask = (volume == label_value)
        labeled_array, num_features = label(mask)
        # Find the largest connected component
        largest_component = (labeled_array == np.argmax(np.bincount(labeled_array.flat)[1:]) + 1)
        refined_volume[largest_component] = label_value
    return refined_volume


def smooth_labels(volume, sigma=1):
    smoothed_volume = np.zeros_like(volume)
    for label_value in np.unique(volume):
        if label_value == 0:
            continue  # Skip background
        mask = (volume == label_value).astype(float)
        smoothed_mask = gaussian_filter(mask, sigma=sigma) > 0.5
        smoothed_volume[smoothed_mask] = label_value
    return smoothed_volume


def load_nifti(file_path):
    nifti_img = nib.load(file_path)
    data = nifti_img.get_fdata()
    return data, nifti_img.affine

def save_nifti(data, affine, output_path):
    data = data.astype(np.int16)
    new_img = nib.Nifti1Image(data, affine)
    nib.save(new_img, output_path)

def morphological_postprocessing(volume, operation="dilation", structure_size=2):
    """
    Apply morphological operations to a labeled 3D segmentation volume.

    Parameters:
    - volume: 3D numpy array of segmented labels [H, W, D].
    - operation: The type of morphological operation to apply ("dilation", "erosion").
    - structure_size: Size of the structuring element (affects how much the operation influences boundaries).
    
    Returns:
    - Refined 3D segmentation volume.
    """
    refined_volume = np.zeros_like(volume)

    # Choose a structuring element suitable for 3D
    struct_elem = ball(structure_size) if volume.ndim == 3 else disk(structure_size)
    
    for label in np.unique(volume):
        if label == 0:  # Skip background label
            continue

        # Create a binary mask for the current label
        binary_mask = (volume == label)
        
        # Apply the specified morphological operation
        if operation == "dilation":
            refined_mask = dilation(binary_mask, struct_elem)
        elif operation == "erosion":
            refined_mask = erosion(binary_mask, struct_elem)
        else:
            raise ValueError("Operation should be 'dilation' or 'erosion'")
        
        # Add the refined mask back to the refined volume
        refined_volume[refined_mask] = label

    return refined_volume

# def main(args):
#     # Load the trained model checkpoint
#     model = torch.load(args.checkpoint)
#     model.eval()

#     # Pick device
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     model.to(device)

#     # Load image and annotation from NIfTI files
#     img, affine = load_nifti(args.image)
#     anno, _ = load_nifti(args.annotation)
    
#     # Use the model to predict labels for the entire 3D volume
#     img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)
#     with torch.no_grad():
#         output = model(img_tensor)
#         output_volume = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
#     # Apply morphological post-processing
#     refined_volume = morphological_postprocessing(output_volume, operation="dilation", structure_size=2)
    
#     # Save the refined volume
#     save_nifti(refined_volume, affine, args.output)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Merging slices parameters')
    # parser.add_argument('--data_folder', type=Path, required=True,
    #                     help="The folder containing the images to predict")
    # parser.add_argument('--source_scan_pattern', type=str,
    #                     help="The pattern to get the original scan. This is used to get the correct metadata")
    # parser.add_argument('--dest_folder', type=Path, required=True)
    # parser.add_argument('--grp_regex', type=str, required=True)

    # parser.add_argument('--num_classes', type=int, default=4)

    parser.add_argument('--image', type=str, default="volumes/segthor/corrected/Patient_13.nii.gz", help="help")
    parser.add_argument('--annotation', type=str, default="data/segthor_train/train/Patient_13/GT_corrected.nii.gz", help="help")
    parser.add_argument('--output', type=str, default="crf/Patient_13_morpho.nii.gz", help="help")
    parser.add_argument('--checkpoint', type=str, default="results/segthor/preprocessed/bestmodel.pkl", help="help")

    args = parser.parse_args()

    print(args)

    return args

def main(args):
    # Load image and annotation from NIfTI files
    img, affine = load_nifti(args.image)  # img is [H, W, D]
    anno, _ = load_nifti(args.annotation)

    # Initialize list to store predictions for each slice
    predictions = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.load(args.checkpoint)
    model.eval().to(device)

    # Process each slice individually
    for i in range(img.shape[2]):
        img_slice = img[:, :, i]
        img_tensor = torch.from_numpy(img_slice).unsqueeze(0).unsqueeze(0).float().to(device)  # Shape [1, 1, H, W]

        with torch.no_grad():
            output = model(img_tensor)
            pred_slice = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # Shape [H, W]
            predictions.append(pred_slice)

    # Stack predictions to form 3D volume
    output_volume = np.stack(predictions, axis=-1)  # Shape [H, W, D]

    # Calculate n_labels using both output_volume and anno
    unique_labels = np.unique(np.concatenate((output_volume, anno)))
    n_labels = len(unique_labels)

    # Apply morphological post-processing or DenseCRF on the entire 3D volume
    refined_volume = morphological_postprocessing(output_volume, operation="dilation", structure_size=5)
    refined_volume = keep_largest_components(refined_volume)
    refined_volume = smooth_labels(refined_volume, sigma=2)
    
    # Save the refined volume
    save_nifti(refined_volume, affine, args.output)

if __name__ == "__main__":
    main(get_args())
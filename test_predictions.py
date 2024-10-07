import argparse
from typing import Any
from pathlib import Path
from pprint import pprint
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from utils import (probs2class, tqdm_, save_images)
from torch.utils.data import DataLoader
from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
import nibabel as nib
from PIL import Image
from tqdm import tqdm

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the clases with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8}

from torch.utils.data.dataloader import default_collate

def custom_collate(batch):
    """
    Custom collate function that handles 'None' ground truth values for the test set.
    If 'gts' is None, we keep the 'images' and 'stems', and set 'gts' to None.
    """
    for sample in batch:
        if sample['gts'] is None:
            # Dummy gts for test set (where we don't have GT predictions)
            sample['gts'] = torch.zeros(1)
    return default_collate(batch)

def save_3d_predictions_as_nii(predictions_3d, output_path, affine=np.eye(4)):
    """
    Save a 3D NumPy array of predictions as a NIfTI file (.nii.gz).

    Parameters:
    - predictions_3d: 3D NumPy array containing the predicted labels.
    - output_path: Path to save the NIfTI file (e.g., "predictions.nii.gz").
    - affine: Optional affine matrix to define the spatial orientation of the NIfTI file.
    """

    nii_img = nib.Nifti1Image(predictions_3d, affine)
    nib.save(nii_img, output_path)

def run_inference_on_test(args):
    # Load the trained model checkpoint
    net = torch.load(args.model_checkpoint)
    net.eval()

    # Pick device
    device = torch.device("cuda") if args.gpu and torch.cuda.is_available() else torch.device("cpu")
    net.to(device)

    # Dataset paths
    root_dir = Path("data") / args.dataset

    # Prepare data loader
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image to the target size
        lambda img: img.convert('L'),  # Convert to grayscale
        lambda img: np.array(img)[np.newaxis, ...],  # Add one dimension to simulate batch
        lambda nd: nd / 255,  # Normalize the image
        lambda nd: torch.tensor(nd, dtype=torch.float32)  # Convert to tensor
    ])

    test_set = SliceDataset('test',
                            root_dir,
                            img_transform=img_transform,
                            debug=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate)

    # Inference
    print(f">> Running inference on the test set...")

    predictions_2d_dir = args.dest / "predictions_2d"
    predictions_2d_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, data in tqdm_(enumerate(test_loader), total=len(test_loader)):
                img = data['images'].to(device)
                stems = data['stems']

                pred_logits = net(img)
                pred_probs = F.softmax(1 * pred_logits, dim=1)
                pred_seg = probs2class(pred_probs)  # Get class predictions

                # Save the 2D prediction for each slice
                for j in range(pred_seg.size(0)):
                        stem = stems[j]
            
                        # Reconstructed image file name
                        patient_id_with_number = '_'.join(stem.split('_')[:2])  # 'Patient_41'
                        slice_idx = stem.split('_')[-1]  # Slice index

                        img_filename = f"{patient_id_with_number}_slice_{slice_idx}.png"

                        # Get prediction
                        slice_pred = pred_seg[j].cpu().numpy()

                        # Convert the prediction to an image and save
                        img = Image.fromarray((slice_pred * 63).astype(np.uint8))  
                        img.save(predictions_2d_dir / img_filename)  


def load_2d_slices(predictions_2d_dir, patient_id):
    """
    Load 2D slices for a given patient and sort them by slice index.
    """
    slice_files = sorted(predictions_2d_dir.glob(f"{patient_id}_slice_*.png"),
                         key=lambda x: int(x.stem.split('_')[-1]))  # Sort by slice index

    slices = [np.array(Image.open(slice_file)) // 63 for slice_file in slice_files]  # Normalize back to original labels
    return slices

def run_3d_reconstruction(args):
    predictions_2d_dir = args.dest / "predictions_2d"
    predictions_3d_dir = args.dest / "predictions_3d_test"
    predictions_3d_dir.mkdir(parents=True, exist_ok=True)

    # Identify unique patient IDs based on the 2D prediction filenames
    patient_ids = {file.stem.split('_slice_')[0] for file in predictions_2d_dir.glob("*.png")}

    for patient_id in tqdm(patient_ids, desc="Reconstructing 3D volumes"):
        # Load all 2D slices for this patient and stack them into a 3D array
        slices = load_2d_slices(predictions_2d_dir, patient_id)
        patient_3d = np.stack(slices, axis=0)  # Stack slices along the Z-axis

        # Save the 3D prediction as .nii.gz
        save_3d_predictions_as_nii(patient_3d, predictions_3d_dir / f"{patient_id}_prediction.nii.gz")
        print(f"Saved 3D prediction for {patient_id}")

    print("3D reconstruction completed.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='SEGTHOR', choices=datasets_params.keys())
    parser.add_argument('--dest', type=Path, required=True, help="Destination directory to save predictions.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--model_checkpoint', type=Path, required=True, help="Path to the model checkpoint for inference.")

    args = parser.parse_args()

    pprint(args)

    # Run inference and save 2D predictions
    run_inference_on_test(args)

    # Reconstruct 3D volumes from 2D predictions
    run_3d_reconstruction(args)


if __name__ == '__main__':
    main()

import argparse
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from pathlib import Path
from typing import Dict
import re



def plot_results(results):
    ids = list(results.keys())
    hausdorff_distances = [results[id_]['hausdorff'] for id_ in ids]
    avg_hausdorff_distances = [results[id_]['avg_hausdorff'] for id_ in ids]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.bar(ids, hausdorff_distances)
    ax1.set_title('Hausdorff Distance per Image')
    ax1.set_xlabel('Image ID')
    ax1.set_ylabel('Hausdorff Distance')
    ax1.tick_params(axis='x', rotation=45)

    ax2.bar(ids, avg_hausdorff_distances)
    ax2.set_title('Average Hausdorff Distance per Image')
    ax2.set_xlabel('Image ID')
    ax2.set_ylabel('Average Hausdorff Distance')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def extract_ground_truth(base_folder: str) -> Dict[str, sitk.Image]:
    base_path: Path = Path(base_folder).resolve()
    pattern = 'Patient_*/GT.nii.gz'
       
    gt_files: list[Path] = []
    gt_files = list(base_path.glob(pattern))
    id_pattern: re.Pattern = re.compile(r"Patient_(\d+)")
    
    results: Dict[str, sitk.Image] = {}
    for gt_file in gt_files:
        match: re.Match | None = id_pattern.search(str(gt_file))
        if match:
            patient_id: str = match.group(1)
            results[patient_id] = sitk.ReadImage(str(gt_file))
    return results


def extract_files(base_folder: str, file_pattern: str, id_pattern: str) -> Dict[str, sitk.Image]:
    base_path: Path = Path(base_folder).resolve()
    files: list[Path] = list(base_path.glob(file_pattern))
    id_regex: re.Pattern = re.compile(id_pattern)
    
    results: Dict[str, sitk.Image] = {}
    for file in files:
        match: re.Match | None = id_regex.search(str(file))
        if match:
            patient_id: str = match.group(1)
            results[patient_id] = sitk.ReadImage(str(file)) 
    return results

# def extract_files(base_folder: str, file_pattern: str, id_pattern: str) -> Dict[str, sitk.Image]:
#     base_path: Path = Path(base_folder).resolve()
#     files: list[Path] = list(base_path.glob(file_pattern))
    
#     id_regex = re.compile(id_pattern)
#     results: Dict[str, sitk.Image] = {}
    
#     for file in files:
#         match = id_regex.search(str(file))
#         if match:
#             patient_id: str = match.group(1)
#             results[patient_id] = sitk.ReadImage(str(file))
    
#     return results
def evaluate_segmentation(ground_truth: sitk.Image, prediction: sitk.Image):
    # Ensure both images have the same size and spacing
    prediction = sitk.Resample(prediction, ground_truth)

    # Calculate Hausdorff distance using SimpleITK
    hausdorff_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_filter.Execute(ground_truth, prediction)
    hausdorff_dist = hausdorff_filter.GetHausdorffDistance()
    avg_hausdorff_dist = hausdorff_filter.GetAverageHausdorffDistance()

    return hausdorff_dist, avg_hausdorff_dist

def run(args):
    base_folder: str = 'data/segthor_train/train'
    ground_truths: Dict[str, sitk.Image] = extract_files(base_folder, "Patient_*/GT.nii.gz", r"Patient_(\d+)")
    # prediction files
    base_folder: str = 'volumes/segthor/UNet/ce'
    predictions: Dict[str, sitk.Image] = extract_files(base_folder, "Patient_*.nii.gz", r"Patient_(\d+)")

    results = {}
    
    for patient_id, pred_image in predictions.items():
       
        if patient_id in ground_truths:
            gt_image = ground_truths[patient_id]
            hausdorff_dist, avg_hausdorff_dist = evaluate_segmentation(gt_image, pred_image)
            results[patient_id] = {
                'hausdorff': hausdorff_dist,
                'avg_hausdorff': avg_hausdorff_dist
            }
            print(f"ID: {patient_id}, Hausdorff: {hausdorff_dist:.4f}, Avg Hausdorff: {avg_hausdorff_dist:.4f}")
        else:
            print(f"Warning: No ground truth file found for ID {patient_id}")

    # Plot results
    # plot_results(results, Path(args.dest_folder) / 'HD.png')

def plot_results(results: Dict[str, Dict[str, float]], save_path: Path):
    # Implementation of plot_results function
    # You'll need to implement this based on your specific plotting needs
        
    # saving figure
    # save_dir = os.path.join('results', 'segthor', 'UNet', 'plots')
    # os.makedirs(save_dir, exist_ok=True)

    # # Saving figure
    # save_path = os.path.join(save_dir, 'HD.png')
    # plt.savefig(save_path)
    # print(f"Figure saved as {save_path}")

    # plt.show()

    pass




def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot Hausdorff distance per image')
    parser.add_argument('--source_scan_pattern', type=str, required=True,
                        help='Pattern for ground truth scans, e.g., "data/segthor_train/train/{id_}/GT.nii.gz"')
    parser.add_argument('--prediction_folder', type=str, required=True,
                        help='Path to the folder containing prediction files') 
    return parser.parse_args()

if __name__ == "__main__":
    run(get_args())
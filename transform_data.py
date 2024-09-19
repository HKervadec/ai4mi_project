import SimpleITK as sitk
import os

def apply_transform_to_segmentation(input_seg_file, transform_file, output_file):
    """
    Apply the BSpline transform to a .nii.gz segmentation and save the result.
    """
    # Read the input segmentation
    input_seg = sitk.ReadImage(input_seg_file)
    
    # Read the transform
    transform = sitk.ReadTransform(transform_file)
    
    # Setup the resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(input_seg)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # NearestNeighbor is usually used for segmentation
    
    # Apply the transform
    transformed_seg = resampler.Execute(input_seg)
    
    # Save the transformed segmentation
    sitk.WriteImage(transformed_seg, output_file)

def main():
    base_folder = 'data/segthor_train/train'
    transform_file = 'data/Slicer BSpline Transform.h5'
    
    # Loop through all patient folders
    for patient_id in range(1, 41):  # Assuming 40 patients
        patient_folder = f'Patient_{patient_id:02d}'
        input_seg_file = os.path.join(base_folder, patient_folder, 'GT.nii.gz')
        output_file = os.path.join(base_folder, patient_folder, 'transformed_segmentation.nii.gz')
        
        # Check if the input segmentation file exists
        if os.path.isfile(input_seg_file):
            print(f'Processing {input_seg_file}...')
            apply_transform_to_segmentation(input_seg_file, transform_file, output_file)
            print(f'Saved transformed segmentation to {output_file}')
        else:
            print(f'Skipping {input_seg_file} (file does not exist)')

if __name__ == '__main__':
    main()







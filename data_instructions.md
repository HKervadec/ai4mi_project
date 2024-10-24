# Data processing instructions

## Data preprocessing

Run `preprocess_augment/preprocessing.py`, with a `data_dir` argument point to your data directory (e.g. data/SEGTHOR).

This will create 2 subfolders in your `data_dir/train` directory, called `img_preprocessed` and `gt_preprocessed`.

## Daa augmentation

Run `preprocess_augment/spatial_augmentation.py` and `preprocess_augment/intensity_augmentation.py`. Again, these also need the `data_dir` argument as above. Additionally, if you want to run the augmentations on the preprocessed data instead of original (preprocessed + augmentation experiment), pass the flag `--run_on_preprocessed` on both scripts.


Each of them will again create 2 subfolders in your `data_dir/train` directory:
- Spatial: `img_spatial_aug` and `gt_spatial_aug` (or `img_pre_spatial_aug` and `gt_pre_spatial_aug` for augmenting the preprocessed data)
- Intensity: `img_intensity_aug` and `gt_intensity_aug` (or `img_pre_intensity_aug` and `gt_pre_intensity_aug` for augmenting the preprocessed data)

## Training

Now that you have the preprocessed + augmented data, you can use it to train models by passing the following arguments with values in `main.py`:
- Preprocessed: `--transformation preprocessed`
- Augmented: `--transformation augmented`
- Preprocessed + augmented:  `--transformation preprocess_augment`

Moreover, to remove the background only images from training pass the flag `--remove_background` to main.py.


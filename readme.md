# AI for medical imaging — Fall 2024 course project
## Group 14

## Project overview
In this README we report the step-by-step to run all our experiments. Remember that in order to run the baseline on a GPU, you just need to run:
`python -O main.py --dataset SEGTHOR --mode full --dest destination_path --gpu`


## 1. Data processing instructions

### 1.1. Data preprocessing

Run `preprocess_augment/preprocessing.py`, with a `data_dir` argument point to your data directory (e.g. data/SEGTHOR).

This will create 2 subfolders in your `data_dir/train` directory, called `img_preprocessed` and `gt_preprocessed`.

### 1.2. Data augmentation

Run `preprocess_augment/spatial_augmentation.py` and `preprocess_augment/intensity_augmentation.py`. Again, these also need the `data_dir` argument as above. Additionally, if you want to run the augmentations on the preprocessed data instead of original (preprocessed + augmentation experiment), pass the flag `--run_on_preprocessed` on both scripts.


Each of them will again create 2 subfolders in your `data_dir/train` directory:
- Spatial: `img_spatial_aug` and `gt_spatial_aug` (or `img_pre_spatial_aug` and `gt_pre_spatial_aug` for augmenting the preprocessed data)
- Intensity: `img_intensity_aug` and `gt_intensity_aug` (or `img_pre_intensity_aug` and `gt_pre_intensity_aug` for augmenting the preprocessed data)

### 1.3. Training

Now that you have the preprocessed + augmented data, you can use it to train models by passing the following arguments with values in `main.py`:
- Preprocessed: `--transformation preprocessed`
- Augmented: `--transformation augmented`
- Preprocessed + augmented:  `--transformation preprocess_augment`

Moreover, to remove the background only images from training pass the flag `--remove_background` to main.py.

## Baseline experiments
### 1. Hyperparameters
We you run the baseline, to modify the hyperparameters you just need to add:
* Epochs `--epochs int_number` (default = 25)
* Learning rate: `--lr float_number` (default = 0.0005)
* Optimizer: `--optimizer option`, where `option = [adam, sgd, adamw] (pick one)`. (default = adam)
* Scheduler: `--scheduler option`, where `option = [None, exp, steps] (pick one)`. (default = None)

### 2. Changes over ENet
* Number of layers `--architecture option`, where `option = [normal, more, less] (pick one)`. (default = normal) 
"more" means the architecture with increased number of layers, "less" is the one with decreased number of layers
* Initial kernel size `--kernelsize int_number` (default = 3)
* K parameter (number of channels) `--channels int_number` (default = 16) 
* Loss 

### 3. DeepLabv3
Instructions

### 4. Post processing
Instructions 

### 5. Evaluation
Instructions

## Test
How to run model inference with test_predictions.py

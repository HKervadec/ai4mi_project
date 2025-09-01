# AI for medical imaging — Fall 2025 group project

## Project overview
The project is based around the SegTHOR challenge data, which was kindly allowed by Caroline Petitjean (challenge organizer) to use for the course. The challenge was originally on the segmentation of different organs: heart, aorta, esophagus and trachea.

### Structure
The project is decomposed in two main parts: weekly individual assignments, and group project:
* [Assignment 01: slicing of the data and 2D, 3D visualization](/weekly_assignments/01_slicing.md);
* [Assignment 02: (to be revealed) something affine](/weekly_assignments/02_affine.md);
* [Assignment 03: Running the baseline, stitching slices to 3D volume and computing metrics](/weekly_assignments/03_train_metrics.md);
* Group project: improve the baseline by adding elements and ideas from the course.

For deadlines and submission guidelines, see the individual assignment pages and the [Canvas page](https://canvas.uva.nl/courses/52878/assignments).

## Codebase features
This codebase is given as a starting point, to provide an initial neural network that converges during training. (For broader context, this is itself a fork of an [older conference tutorial](https://github.com/LIVIAETS/miccai_weakly_supervised_tutorial) we gave few years ago.) It also provides facilities to locally run some test on a laptop, with a toy dataset and dummy network.

Summary of codebase (in PyTorch)
* slicing the 3D Nifti files to 2D `.png`; **To implement as assignment 01**
* stitching 2D `.png` slices to 3D volume compatible with initial nifti files; **To implement as assignment 03**
* basic 2D segmentation network;
* basic training and printing with cross-entroly loss and Adam;
* partial cross-entropy alternative as a loss (to disable one class during training);
* debug options and facilities (cpu version, "dummy" network, smaller datasets);
* saving of predictions as `.png`;
* log the 2D DSC and cross-entropy over time, with basic plotting;
* tool to compare different segmentations (`viewer/viewer.py`).

**Some recurrent questions might be addressed here directly.** As such, it is expected that small change or additions to this readme to be made.

## Codebase use
In the following, a line starting by `$` usually means it is meant to be typed in the terminal (bash, zsh, fish, ...), whereas no symbol might indicate some python code.

### Setting up the environment
```
$ git clone https://github.com/HKervadec/ai4mi_project.git
$ cd ai4mi_project
$ git submodule init
$ git submodule update
```

This codebase was written for a somewhat recent python (3.10 or more recent). (**Note: Ubuntu and some other Linux distributions might make the distasteful choice to have `python` pointing to 2.+ version, and require to type `python3` explicitly.**) The required packages are listed in [`requirements.txt`](requirements.txt) and a [virtual environment](https://docs.python.org/3/library/venv.html) can easily be created from it through [pip](https://pypi.org/):
```
$ python -m venv ai4mi
$ source ai4mi/bin/activate
$ which python  # ensure this is not your system's python anymore
$ python -m pip install -r requirements.txt
```
Conda is an alternative to pip, but is recommended not to mix `conda install` and `pip install`.

### Setting up the environment - Some troubleshooting for windows users 
These steps assume you are using Git Bash + Anaconda + an IDE (e.g., PyCharm).

Open git bash and run:
Step 1:
```
$ git clone https://github.com/HKervadec/ai4mi_project.git
$ cd ai4mi_project
$ git submodule init
$ git submodule update
```
Step 2:
```
# 1) Create a fresh conda env with Python 3.10+ (matches project note)
conda create -n ai4mi python=3.10 -y

# 2) Activate it
conda activate ai4mi

# 3) (Optional but nice) make sure pip is present/updated
python -m pip install --upgrade pip

# 4) From the repo folder, install dependencies with pip
python -m pip install -r requirements.txt
```

Some common troubleshooting for windows users:

In case in bash u got - conda: command not found

Open Anaconda Prompt:
```
conda init bash

#Find where conda is installed
where conda
```
Yous should get sth like - C:\Users\<YourName>\anaconda3

Close and open git bash - change CONDA_HOME in the code below

```
$ CONDA_HOME="/c/Users/<YourName>/anaconda3"
if [ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
    . "$CONDA_HOME/etc/profile.d/conda.sh"
else
    export PATH="$CONDA_HOME:$CONDA_HOME/Scripts:$CONDA_HOME/Library/bin:$PATH"
fi
```

### Getting the data
The synthetic dataset is generated randomly, whereas for Segthor it is required to put the file [`segthor_train.zip`](https://amsuni-my.sharepoint.com/:u:/g/personal/h_t_g_kervadec_uva_nl/EfMdFte7pExAnPwt4tYUcxcBbJJO8dqxJP9r-5pm9M_ARw?e=ZNdjee) (required a UvA account) in the `data/` folder. If the computer running it is powerful enough, the recipe for `data/SEGTHOR` can be modified in the [Makefile](Makefile) to enable multi-processing (`-p -1` option, see `python slice_segthor.py --help` or its code directly).
```
$ make data/TOY2
$ make data/SEGTHOR
```

**Implementing slice_segthor.py is part of assignment 01**. An official implementation will be shared by assignment 02.

For windows users, you can use the following instead
```
$ rm -rf data/TOY2_tmp data/TOY2
$ python gen_two_circles.py --dest data/TOY2_tmp -n 1000 100 -r 25 -wh 256 256
$ mv data/TOY2_tmp data/TOY2

$ sha256sum -c data/segthor_train.sha256
$ unzip -q data/segthor_train.zip

$ rm -rf data/SEGTHOR_tmp data/SEGTHOR
$ python  slice_segthor.py --source_dir data/segthor_train --dest_dir data/SEGTHOR_tmp \
         --shape 256 256 --retain 10
$ mv data/SEGTHOR_tmp data/SEGTHOR
````

### Viewing the data
The data can be viewed in different ways:
- looking directly at the `.png` in the sliced folder (`data/TOY2`, `data/SEGTHOR`);
- using the provided "viewer" to compare segmentations ([see below](#viewing-the-results));
- opening the Nifti files from `data/segthor_train` with [3D Slicer](https://www.slicer.org/) or [ITK Snap](http://www.itksnap.org).

#### 2D viewer
Comparing some predictions with the provided [viewer](viewer/viewer.py) (right-click to go to the next set of images, left-click to go back), or simply looking at the data:
```
$ python viewer/viewer.py --img_source data/TOY2/val/img \
    data/TOY2/val/gt \
    --show_img -C 256 --no_contour
```
![Example of the viewer on the TOY example](viewer_toy.png)
**Note:** if using it from a SSH session, it requires X to be forwarded ([Unix/BSD](https://man.archlinux.org/man/ssh.1#X), [Windows](https://mobaxterm.mobatek.net/documentation.html#1_4)) for it to work. Note that X forwarding also needs to be enabled on the server side.


#### 3D viewers
[3D Slicer](https://www.slicer.org/) and [ITK Snap](http://www.itksnap.org) are two popular viewers for medical data. We recommend 3D Slicer for the course.


## Known issues
### Cannot pickle lambda in the dataloader
Some installs (probably due to Python/Pytorch version mismatch) throw an error about an inability to pickle lambda functions (at the dataloader stage). Short of reinstalling everything, setting the number of workers to 0 seems to get around the problem (`--num_workers 0`).

### Pytorch not compiled for Numpy 2.0
It may happen that Pytorch, when installed through pip, was compiled for Numpy 1.x, which creates some inconsistencies. Downgrading Numpy seems to solve it: `pip install --upgrade "numpy<2"`

### Viewer on Windows
Windows has different paths names (`\` in stead of `/`), so the default regex in the viewer needs to be changed to `--id_regex=".*\\\\(.*).png"`.

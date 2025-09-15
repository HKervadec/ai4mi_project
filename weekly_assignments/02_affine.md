# AI for medical imaging: individual assignment 2
## Matrices and dots
The goal of this assignment is to practice image transformation from a known matrix for rigid transformations.

### Overview

The Segthor dataset you have been provided features the 3D CT images in NIfTI format as well as the annotations for some of the anatomical structures.

You may have noticed that the annotations for the heart class are not aligned with the heart in the CT scan, which should be apparent even without in-depth knowledge of the human anatomy. The annotations have been tampered with, and your task will be to fix them.

### Useful information

You are provided the affine matrices that have been used to tamper the heart annotations. 

$`T_1 = \begin{bmatrix}
    1 & 0 & 0 & 275 \\
    0 & 1 & 0 & 200  \\
    0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 1
\end{bmatrix}`$

$`R_2 =
\begin{bmatrix}
np.cos(ϕ) & -np.sin(ϕ) & 0 & 0 \\
np.sin(ϕ) & np.cos(ϕ) & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1  \\
\end{bmatrix}`$ with  $`\phi=-(27/180)\pi`$

$`T_3 = {T_1}^{-1}`$

$`T_4 =
\begin{bmatrix}
1 & 0 & 0 & 50 \\
0 & 1 & 0 & 40 \\
0 & 0 & 1 & 15 \\
0 & 0 & 0 &  1
\end{bmatrix}`$

They were applied in this order: $`T_1`$, then $`R_2`$, then $`T_3`$, then $`T_4`$

To guide you in your search for a solution, you may rely on the following:

* Patient 27 has two sets of annotations: one has been tampered with, the other hasn't. All the other patients only have tampered annotations.
* Only the _heart_ annotation has been tampered with.
* All files have been tampered with the exact same process.
* To tamper the _heart_ annotations, the affine matrices have been applied using `scipy.ndimage.affine_transform`, for which you may find the related docs [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html).


### The task

You are to create a script that fixes the tampered annotations. The script will be tested by running it with the following command:

```
python your_script.py --source_dir /absolute/path/to/folder/
```

The source directory will containing the same data you are working with, and will have the same structure:
```
└─ data/   <-- the source_dir argument
   ├─ Patient_01/
   │   ├─ GT.nii.gz
   │   └─ Patient_01.nii.gz
   ├ Patient_02/
   │   ├─ GT.nii.gz
   │   └─ Patient_02.nii.gz
   ...
   └─ Patient_40/
       ├─ GT.nii.gz
       └─ Patient_40.nii.gz
```

The script should only process the files named `GT.nii.gz`, and save the results in the same folder as the source with the filename `GT_fixed.nii.gz`
```
└─ data/   <-- the source_dir argument
   ├─ Patient_01/
   │   ├─ GT.nii.gz
   │   ├─ GT_fixed.nii.gz  <-- your result
   │   └─ Patient_01.nii.gz
   ├─ Patient_02/
   ...
```

Please submit a single python file.

### Contact

Pierandrea Cancian (<p.cancian@uva.nl>) (lead), Mohammad Islam (<m.m.islam@uva.nl>) (helper).
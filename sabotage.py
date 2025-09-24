#!/usr/bin/env python3.10

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
from pathlib import Path
from pprint import pprint
from functools import partial
from multiprocessing import Pool
from shutil import copy

import numpy as np
import scipy as sp
import nibabel as nib
from numpy import pi as π

from utils import tqdm_

ID = np.diag([1, 1, 1, 1])
TR = np.asarray([[1, 0, 0, 50],
                 [0, 1, 0, 40],  # noqa: E241
                 [0, 0, 1, 15],  # noqa: E241
                 [0, 0, 0, 1]])  # noqa: E241

DEG: int = 27
ϕ: float = - DEG / 180 * π
RO = np.asarray([[np.cos(ϕ), -np.sin(ϕ), 0, 0],  # noqa: E241, E201
                 [np.sin(ϕ),  np.cos(ϕ), 0, 0],  # noqa: E241
                 [     0,         0,     1, 0],  # noqa: E241, E201
                 [     0,         0,     0, 1]])  # noqa: E241, E201

X_bar: float = 275
Y_bar: float = 200
Z_bar: float = 0
C1 = np.asarray([[1, 0, 0, X_bar],
                 [0, 1, 0, Y_bar],
                 [0, 0, 1, Z_bar],
                 [0, 0, 0,    1]])  # noqa: E241
C2 = np.linalg.inv(C1)

AFF = C1 @ RO @ C2 @ TR
INV = np.linalg.inv(AFF)
print(f"{AFF=}")
print(f"{RO=}")
print(f"{AFF=}")
print(f"{INV=}")


def transform(file: Path, source: Path, dest: Path, K: int, affine: np.ndarray,
              discard_orig_aff: bool) -> None:
        nib_obj = nib.load(file)

        ct_orig: np.ndarray = np.asarray(nib_obj.dataobj)
        ct_new: np.ndarray = ct_orig[...]

        isolated_class: np.ndarray = ct_orig == K
        ct_new[isolated_class] = 0

        transformed_: np.ndarray = sp.ndimage.affine_transform(isolated_class,
                                                               matrix=affine,
                                                               order=0)

        if not set(np.unique(transformed_)) == {False, True}:
                print(file, np.unique(transformed_))
        ct_new[transformed_] = K
        assert ct_new.shape == ct_orig.shape
        assert (ct_orig == K).sum() == (ct_new == K).sum()

        for k in range(1, len(ct_new)):
                if k == K:
                        continue

                assert np.array_equal((ct_orig == k), (ct_new == k))

        tr_nib = nib.nifti1.Nifti1Image(ct_new,
                                        affine=np.diag([1, 1, 1, 1]) if discard_orig_aff else nib_obj.affine,
                                        header=nib_obj.header)
        dest_file: Path = dest / file.relative_to(source)
        dest_file.parent.mkdir(exist_ok=True, parents=True)
        nib.save(tr_nib, dest_file)


def dispatch(file: Path, source: Path, dest: Path, regex: str, K: int, affine: np.ndarray, discard_orig_aff: bool) -> None:
        if file.match(regex):
                transform(file, source, dest, K, affine, discard_orig_aff)
        else:
                dest_file: Path = dest / file.relative_to(source)
                dest_file.parent.mkdir(exist_ok=True, parents=True)
                copy(file, dest_file)


def main(args: argparse.Namespace) -> None:
        source_niftis: list[Path] = list(args.source_dir.glob("**/*.nii.gz"))
        pprint(source_niftis)

        args.dest_dir.mkdir(exist_ok=True, parents=True)

        iterator = tqdm_(source_niftis)
        affine: np.ndarray
        match args.mode:
                case 'normal':
                        affine = AFF
                case 'inv' | 'inverse':
                        affine = INV
                case 'id':
                        affine = ID

        fn = partial(dispatch,
                     source=args.source_dir,
                     dest=args.dest_dir,
                     regex=args.regex_gt,
                     K=args.K,
                     affine=affine,
                     discard_orig_aff=args.discard_orig_aff)
        match args.process:
                case 1:
                        for f in iterator:
                                fn(f)
                case -1:
                        Pool().map(fn, iterator)
                case _ as p:
                        Pool(p).map(fn, iterator)


def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='''
                Sabotage params. Will copy all nifti from source to dest dir
                (including the scans),
                and modify on the fly the identified ground truth files.''')
        parser.add_argument('--source_dir', type=Path, required=True)
        parser.add_argument('--dest_dir', type=Path, required=True)
        parser.add_argument('-K', '-C', type=int, required=True,
                            help="The class to modify.")
        parser.add_argument('--regex_gt', type=str, required=True,
                            help="The regex to identify the ground truth files.")
        parser.add_argument('--process', '-p', type=int, default=1,
                            help="The number of cores to use for processing")

        parser.add_argument('--mode', type=str, choices=['normal', 'inv', 'inverse', 'id'], default='normal')
        parser.add_argument('--discard_orig_aff', action='store_true')

        args = parser.parse_args()

        print(args)

        return args


if __name__ == "__main__":
        main(get_args())

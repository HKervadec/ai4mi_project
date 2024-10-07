import zipfile
from torch.utils.data import Dataset
import os
import logging

from typing import Any
from collections import deque
import numpy as np
import torch
import scipy.sparse as sp


class VolumetricDataset(Dataset):
    """
    A PyTorch Dataset class for handling volumetric data.
    Attributes:
        path (str): Path to the dataset directory.
        ratio (float): Ratio of the dataset to be used for training.
        processor (Any): An object responsible for preprocessing and transforming the data.
        num_classes (int): Number of classes in the dataset.
        train (bool): Flag indicating whether the dataset is for training or testing. Default is True.
        cache_size (int): Size of the cache for storing preprocessed items. Default is 0.
    Methods:
        __init__(path: str, ratio: float, processor: Any, num_classes: int, train: bool = True, cache_size = 0):
            Initializes the dataset with the given parameters.
        _prepare_data():
            Prepares the dataset by listing all IDs and splitting them based on the ratio.
        __len__():
            Returns the length of the dataset.
        _load_item(idx):
            Loads and preprocesses a single item from the dataset.
        __getitem__(idx):
            Retrieves an item from the dataset, using cache if enabled.
    """

    def __init__(
        self,
        path: str,
        ratio: float,
        processor: Any,
        num_classes: int,
        train: bool = True,
        cache_size=0,
    ):
        """
        Initializes the dataset object.
        Args:
            path (str): The path to the dataset.
            ratio (float): The train/test ratio of the dataset to be used.
            processor (Any): The processor to be used for data processing.
            num_classes (int): The number of classes in the dataset.
            train (bool, optional): Flag indicating whether the dataset is for training. Defaults to True.
            cache_size (int, optional): The number of images to keep in memory. Defaults to 0.
        """

        self.path = path
        self.ratio = ratio

        self.processor = processor
        self.train = train
        self.num_classes = num_classes
        self.cache_size = cache_size

        self._prepare_data()

    def _prepare_data(self):
        self.ids = sorted(
            filter(
                lambda dir: os.path.isdir(os.path.join(self.path, dir)),
                os.listdir(self.path),
            )
        )
        if self.train:
            self.data = self.ids[: int(len(self.ids) * self.ratio)]
        else:
            self.data = self.ids[int(len(self.ids) * self.ratio) :]

        if self.cache_size:
            self.queue = deque(maxlen=self.cache_size)

    def __len__(self):
        return len(self.data)

    def _load_item(self, idx):
        npy_path = os.path.join(self.path, self.data[idx], f"{self.data[idx]}.npy")
        if os.path.exists(npy_path):
            ct = np.load(npy_path)
            try:
                gt = sp.load_npz(npy_path.replace(".npy", "_gt.npz"))
                gt = gt.toarray().reshape((self.num_classes, *ct.shape[1:]))
                ct, gt = torch.from_numpy(ct), torch.from_numpy(gt)
            except zipfile.BadZipFile:
                # Don't ask.
                logging.error("Error happened. File is not a zip file. Reloading GT.")
                ct_path = os.path.join(self.path, self.data[idx], f"{self.data[idx]}.nii.gz")
                gt_path = os.path.join(self.path, self.data[idx], "GT.nii.gz")

                ct, gt = self.processor.preprocess_ct_gt(ct_path, gt_path, self.num_classes)
                # Ground truth is extra compressed to save space
                np.save(npy_path, ct)
                sp.save_npz(npy_path.replace(".npy", "_gt.npz"), sp.csr_array(gt.flatten()))

    
        else:
            ct_path = os.path.join(
                self.path, self.data[idx], f"{self.data[idx]}.nii.gz"
            )
            gt_path = os.path.join(self.path, self.data[idx], "GT.nii.gz")

            ct, gt = self.processor.preprocess_ct_gt(ct_path, gt_path, self.num_classes)
            # Ground truth is extra compressed to save space
            np.save(npy_path, ct)
            sp.save_npz(npy_path.replace(".npy", "_gt.npz"), sp.csr_array(gt.flatten()))

        if self.train:
            data_item = self.processor.train_transform(ct, gt)

        else:
            data_item = self.processor.zoom_transform(ct, gt)

        return (idx, data_item)

    def __getitem__(self, idx):
        """
        Retrieve an item from the dataset at the specified index.
        If caching is enabled, the method first checks if the item is in the cache.
        If found, it returns the cached item. If not found, it loads the item,
        caches it, and then returns it.
        If caching is not enabled, the method directly loads and returns the item.
        Args:
            idx (int): The index of the item to retrieve.
        Returns:
            The data item corresponding to the specified index.
        """
        if self.cache_size:
            for i, item in self.queue:
                if i == idx:
                    return item
                
            # Otherwise find and append it to the list
            item = self._load_item(idx)  # (idx, data)
            self.queue.append(item)
            return item[1]

        else:
            return self._load_item(idx)[1]

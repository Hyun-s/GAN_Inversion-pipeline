# python3.7
"""Contains the class of dataset."""

import os
import pickle
import string
import zipfile
import numpy as np
import cv2
import lmdb

import torch
from torch.utils.data import Dataset

from .transforms import progressive_resize_image
from .transforms import crop_resize_image
from .transforms import resize_image
from .transforms import normalize_image

__all__ = ['BaseDataset']

_FORMATS_ALLOWED = ['dir', 'lmdb', 'list', 'zip']


class ZipLoader(object):
    """Defines a class to load zip file.

    This is a static class, which is used to solve the problem that different
    data workers can not share the same memory.
    """
    files = dict()

    @staticmethod
    def get_zipfile(file_path):
        """Fetches a zip file."""
        zip_files = ZipLoader.files
        if file_path not in zip_files:
            zip_files[file_path] = zipfile.ZipFile(file_path, 'r')
        return zip_files[file_path]

    @staticmethod
    def get_image(file_path, image_path):
        """Decodes an image from a particular zip file."""
        zip_file = ZipLoader.get_zipfile(file_path)
        image_str = zip_file.read(image_path)
        image_np = np.frombuffer(image_str, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        return image


class LmdbLoader(object):
    """Defines a class to load lmdb file.

    This is a static class, which is used to solve lmdb loading error
    when num_workers > 0
    """
    files = dict()

    @staticmethod
    def get_lmdbfile(file_path):
        """Fetches a lmdb file"""
        lmdb_files = LmdbLoader.files
        if 'env' not in lmdb_files:
            env = lmdb.open(file_path,
                            max_readers=1,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            with env.begin(write=False) as txn:
                num_samples = txn.stat()['entries']
            cache_file = '_cache_' + ''.join(
                c for c in file_path if c in string.ascii_letters)
            if os.path.isfile(cache_file):
                keys = pickle.load(open(cache_file, "rb"))
            else:
                with env.begin(write=False) as txn:
                    keys = [key for key, _ in txn.cursor()]
                pickle.dump(keys, open(cache_file, "wb"))
            lmdb_files['env'] = env
            lmdb_files['num_samples'] = num_samples
            lmdb_files['keys'] = keys
        return lmdb_files

    @staticmethod
    def get_image(file_path, idx):
        """Decodes an image from a particular lmdb file"""
        lmdb_files = LmdbLoader.get_lmdbfile(file_path)
        env = lmdb_files['env']
        keys = lmdb_files['keys']
        with env.begin(write=False) as txn:
            imagebuf = txn.get(keys[idx])
        image_np = np.frombuffer(imagebuf, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        return image


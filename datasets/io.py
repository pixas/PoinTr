import h5py
import numpy as np
import open3d
import os
import json
import logging

import torch

try:
    from petrel_client.client import Client
except ImportError:
    # raise ImportError('Please install petrel_client')
    logging.warning('Please install petrel_client''Please install petrel_client')

import io


class CephManager:

    def __init__(self, s2_conf_path='~/petreloss.conf'):
        self.conf_path = s2_conf_path
        self._client = Client(conf_path=s2_conf_path)

    def readlines(self, url):

        response = self._client.get(url, enable_stream=True, no_cache=True)

        lines = []
        for line in response.iter_lines():
            lines.append(line.decode('utf-8'))
        return lines

    def load_data(self, path, ceph_read=False):
        if ceph_read:
            return self.readlines(path)
        else:
            return self._client.get(path)

    def get(self, file_path):
        return self._client.get(file_path)


    def load_json(self, json_url):
        return json.loads(self.load_data(json_url, ceph_read=False))

    def load_model(self, model_path, map_location):
        file_bytes = self._client.get(model_path)
        buffer = io.BytesIO(file_bytes)
        return torch.load(buffer, map_location=map_location)

    def write(self, save_dir, obj):
        self._client.put(save_dir, obj)

    def put_text(self,
                 obj: str,
                 filepath,
                 encoding: str = 'utf-8') -> None:
        self.write(filepath, bytes(obj, encoding=encoding))

    def exists(self, url):
        return self._client.contains(url)
    
    def remove(self, url):
        return self._client.delete(url)
    
    def isdir(self, url):
        return self._client.isdir(url)

    def isfile(self, url):
        return self.exists(url) and not self.isdir(url)

    def listdir(self, url):
        return self._client.list(url)

    def copy(self, src_path, dst_path, overwrite):
        if not overwrite and self.exists(dst_path):
            pass
        object = self._client.get(src_path)
        self._client.put(dst_path, object)
        return dst_path

class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)
       
    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!
    @classmethod
    def _read_pcd(cls, file_path):
        pc = open3d.io.read_point_cloud(file_path)
        ptcloud = np.array(pc.points)
        
        return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]
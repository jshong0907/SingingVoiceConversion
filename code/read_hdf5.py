import h5py
import numpy as np
from torch.utils import data


class ReadHDF5(data.Dataset): # dataset을 읽기 위한 클래스
    def __init__(self, file_name, key): # 생성자
        self.hf = h5py.File(file_name, 'r')
        self.data = self.hf.get(key).value
        self.len = self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len
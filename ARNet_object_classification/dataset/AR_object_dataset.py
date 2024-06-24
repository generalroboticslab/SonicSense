import os
import sys
import torch
import numpy as np
from random import *
from natsort import natsorted
import pytorch_lightning as pl
sys.path.append('.')

class AR_object_dataset(pl.LightningDataModule):
    def __init__(self, split,params):
        assert split in ['train', 'valid', 'test'], "split error value!"
        self.params=params
        self.split = split
        self.current_label_list=params.label_mapping
        self.contact_points_paths,self.audio_paths, self.labels = self._load_data()
        print(split,len(self.contact_points_paths),len(self.audio_paths))
        self.total_num_of_data=len(self.contact_points_paths)

    def __getitem__(self, index):
        labels =np.array(self.labels[index])
        contact_points_path=self.contact_points_paths[index]
        audio_path=self.audio_paths[index]
        contact_points= self.read_contact_points_from_npy(contact_points_path)
        audio = self.read_audio_from_npy(audio_path)
        self.mean, self.std = np.asarray(audio).mean(),np.asarray(audio).std()
        audio=(audio-self.mean)/self.std
        return torch.from_numpy(contact_points), torch.from_numpy(audio), torch.from_numpy(labels)

    def __len__(self):
        return len(self.contact_points_paths)

    def _load_data(self):
        contact_points_paths = []
        audio_paths = []
        label = []
        for file_name in natsorted(os.listdir(f'./{self.params.exp_name}/contact_points/{self.split}')):
            contact_points_paths.append(f'./{self.params.exp_name}/contact_points/{self.split}/{file_name}')
            for label_name in self.params.label_mapping:
                if f'_{label_name}.npy' in file_name:
                    label.append(self.params.label_mapping[label_name])
        for path in contact_points_paths:
            path = path.replace('contact_points','audio')
            audio_paths.append(path)
        return contact_points_paths, audio_paths, label
    
    def read_contact_points_from_npy(self, path):
        raw_contact_points = np.load(f'{path}', allow_pickle=True)
        contact_points = []
        for i in raw_contact_points:
            contact_points.append(i[0:3])
        return np.array(contact_points, np.float32)

    def read_audio_from_npy(self, path):
        audio_data = np.load(f'{path}', allow_pickle=True)
        return np.array(audio_data, np.float32)

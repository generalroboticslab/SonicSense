import re
import sys
import json
import torch
import librosa
import numpy as np
from random import *
import open3d as o3d
import pytorch_lightning as pl
import matplotlib.pyplot as plt
sys.path.append('.')

class AR_material_dataset(pl.LightningDataModule):
    def __init__(self, split, params):
        assert split in ['train', 'valid', 'test','original_valid','original_test'], "split error value!"
        self.params=params
        self.split = split
        self.exp=params.exp_name
        self.gaussian_scale=params.gaussian_scale
        self.current_label_list=params.label_mapping

        with open('dataset/material_simple_categories.json') as f:
            self.simplified_label_mapping = json.load(f)

        self.material_label_paths,self.audio_paths = self._load_data(load_from_npy=False)
        print(split,len(self.material_label_paths),len(self.audio_paths))
        self.total_num_of_data=len(self.material_label_paths)

    def __getitem__(self, index):
        labels = np.array(self.current_label_list[int(self.read_simplified_label_from_npy(self.material_label_paths[index]))])
        audio_path=self.audio_paths[index][0].replace('.wav','.npy')
        audio = self.read_audio_from_npy([audio_path])
        #normalize audio data
        self.mean, self.std = np.asarray(audio).mean(),np.asarray(audio).std()
        audio=(audio-self.mean)/self.std
        if self.params.display_audio == True:
            librosa.display.specshow(audio[0], x_axis='time', y_axis='mel',sr=44100,cmap='inferno')
            plt.colorbar(cmap='inferno',format='%+2.f')
            plt.savefig(f'images/{index}',dpi=300)
            plt.close()
        return torch.from_numpy(audio), torch.from_numpy(labels)

    def __len__(self):
        return len(self.material_label_paths)

    def _load_data(self,load_from_npy=True):
        material_label_paths=np.load(f'data/ARdataset/{self.exp}/{self.split}_label_list.npy', allow_pickle=True)
        audio_paths=np.load(f'data/ARdataset/{self.exp}/{self.split}_audio_list.npy', allow_pickle=True)
        labels=[0]*len(self.current_label_list)
        
        #put the label filepath and audio filepath in its own class and count the number
        for i in range(len(material_label_paths)):
            label = self.current_label_list[int(self.read_simplified_label_from_npy(material_label_paths[i]))]
            labels[label]+=1 
        print(f'number of {self.split} data =',len(material_label_paths) )
        for i in range(len(labels)):
            print(f'{self.split}_label_{i} % =',labels[i]/len(material_label_paths))
        print(self.split,'==========================total data:',len(material_label_paths))
        return material_label_paths, audio_paths
    
    def get_mean_and_std_of_the_audio_data(self):
        audio_path=np.load(f'data/ARdataset/{self.exp}/{self.split}_audio_list.npy', allow_pickle=True)
        data=[]
        for path in audio_path:
            path=[path[0].replace('.wav','.npy')]
            npy=self.read_audio_from_npy(path)
            data.append(npy)
        print('mean and std of the dataset', np.asarray(data).mean(),np.asarray(data).std())
        return np.asarray(data).mean(),np.asarray(data).std()
    
    def data_augmentation_techniques(self):
        if self.split == 'train':
            #frequency masking
            starting_frequencey=randint(0,60)
            print('check_random',starting_frequencey)
            for i in range(starting_frequencey,starting_frequencey+4):
                audio[0][i]=audio[0][i]*0
            #time masking
            starting_time=randint(0,60)
            for i in range(len(audio[0])):
                for j in range(starting_time,starting_time+4):             
                    audio[0][i][j]=0
            #gaussian noise
                gaussian = np.random.normal(self.mean, self.gaussian_scale, (64, 64))
                gaussian = np.array([gaussian], np.float32)
                audio = audio + gaussian

    def get_index_from_audiofilename(self,filename):
        number=re.findall(r'\d+', filename)
        return [int(x) for x in number]
    def read_point_cloud_from_txt(self, path):
        data = []
        with open(f'{path}', "r") as f:
            for line in f:
                ls = line.strip().split()
                ls = [float(i) for i in ls]
                data.append(ls)
        points = []
        for i in range(len(data)):
            if data[i][4] == 1:
                points.append(data[i][0:3])

            else:
                points.append([0, 0, 0])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=float))
        pcd.scale(0.01, [0, 0, 0])

    def read_contact_from_txt(self, path):
        path, idx = path
        contact = []

        with open(f'{path}', "r") as f:
            for line in f:
                ls = line.strip().split()
                ls = [float(i) for i in ls]
                contact.append(ls[4])
        return contact[idx]
    
    def read_label_from_npy(self, path):
        path, idx = path
        idx=idx[1]+(idx[0]-1)*4
        data = [np.load(f'{path}', allow_pickle=True)[idx]]
        label=data[0]
        return np.array(label)
    
    def read_simplified_label_from_npy(self, path):
        path, idx = path
        if 'objectfolder' in path:
            label=1
        else:
            idx=idx[1]+(idx[0]-1)*4
            data = [np.load(f'{path}', allow_pickle=True)[idx]]
            label=data[0]
        for key in self.simplified_label_mapping:
            if label in self.simplified_label_mapping[key]:
                simplified_label= list(self.simplified_label_mapping.keys()).index(key)

        return np.array(int(simplified_label))
    def read_audio_from_npy(self, path):
        path = path[0]
        audio_data = np.load(f'{path}', allow_pickle=True)
        return np.array([audio_data], np.float32)



class AR_material_objectfolder_dataset(pl.LightningDataModule):
    def __init__(self, split,params):
        assert split in ['train', 'valid', 'test','original_valid','original_test'], "split error value!"
        self.params=params
        self.split = split
        self.exp=params.exp_name
        self.gaussian_scale=params.gaussian_scale
        self.current_label_list=params.label_mapping

        with open('dataset/material_simple_categories.json') as f:
            self.simplified_label_mapping = json.load(f)
        ###    #0,3,4,6,8 corresponding to orignial material label0 3 4 6 9; 
        self.object_folder_label_mapping={0:0,3:1,4:2,6:3,8:4}
        self.object_folder_label_mapping

        self.material_label_paths,self.audio_paths = self._load_data(load_from_npy=False)
        print(split,len(self.material_label_paths),len(self.audio_paths))
        self.total_num_of_data=len(self.material_label_paths)

    def __getitem__(self, index):
        labels = np.array(self.object_folder_label_mapping[self.current_label_list[int(self.read_simplified_label_from_npy(self.material_label_paths[index]))]])
        audio_path=self.audio_paths[index][0].replace('.wav','.npy')
        audio = self.read_audio_from_npy([audio_path])
        #normalize audio data
        self.mean, self.std = np.asarray(audio).mean(),np.asarray(audio).std()
        audio=(audio-self.mean)/self.std
        if self.params.display_audio == True:
            librosa.display.specshow(audio[0], x_axis='time', y_axis='mel',sr=44100,cmap='inferno')
            plt.colorbar(cmap='inferno',format='%+2.f')
            plt.savefig(f'images/{index}',dpi=300)
            plt.close()
        return torch.from_numpy(audio), torch.from_numpy(labels)

    def __len__(self):
        return len(self.material_label_paths)

    def _load_data(self,load_from_npy=True):
        material_label_paths=np.load(f'data/ARdataset/{self.exp}/{self.split}_label_list.npy', allow_pickle=True)
        audio_paths=np.load(f'data/ARdataset/{self.exp}/{self.split}_audio_list.npy', allow_pickle=True)
        labels=[0]*len(self.current_label_list)
        
        material_label_list_for_objectfolder=[]
        audio_path_list_for_objectfolder=[]
        for idx in range(len(material_label_paths)):
            label = np.array(self.current_label_list[int(self.read_simplified_label_from_npy(material_label_paths[idx]))])
            if label in [0, 3, 4, 6, 8]:
                material_label_list_for_objectfolder.append(material_label_paths[idx])
                audio_path_list_for_objectfolder.append(audio_paths[idx])

        if self.split == 'test':
            material_label_paths=np.array(material_label_list_for_objectfolder)
            audio_paths = np.array(audio_path_list_for_objectfolder)

        #put the label filepath and audio filepath in its own class and count the number
        for i in range(len(material_label_paths)):
            label = self.current_label_list[int(self.read_simplified_label_from_npy(material_label_paths[i]))]
            labels[label]+=1 
        print(f'number of {self.split} data =',len(material_label_paths) )
        for i in range(len(labels)):
            print(f'{self.split}_label_{i} % =',labels[i]/len(material_label_paths))
        print(self.split,'==========================total data:',len(material_label_paths))
        return material_label_paths, audio_paths
   
    def get_index_from_audiofilename(self,filename):
        number=re.findall(r'\d+', filename)
        return [int(x) for x in number]
    
    def read_point_cloud_from_txt(self, path):
        data = []
        with open(f'{path}', "r") as f:
            for line in f:
                ls = line.strip().split()
                ls = [float(i) for i in ls]
                data.append(ls)
        points = []
        for i in range(len(data)):
            if data[i][4] == 1:
                points.append(data[i][0:3])
            else:
                points.append([0, 0, 0])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=float))
        pcd.scale(0.01, [0, 0, 0])

    def read_contact_from_txt(self, path):
        path, idx = path
        contact = []
        with open(f'{path}', "r") as f:
            for line in f:
                ls = line.strip().split()
                ls = [float(i) for i in ls]
                contact.append(ls[4])
        return contact[idx]
    
    def read_label_from_npy(self, path):
        path, idx = path
        idx=idx[1]+(idx[0]-1)*4
        data = [np.load(f'{path}', allow_pickle=True)[idx]]
        label=data[0]
        return np.array(label)
    
    def read_simplified_label_from_npy(self, path):
        path, idx = path
        if 'objectfolder' in path:
            label=1
        else:
            idx=idx[1]+(idx[0]-1)*4
            data = [np.load(f'{path}', allow_pickle=True)[idx]]
            label=data[0]
        for key in self.simplified_label_mapping:
            if label in self.simplified_label_mapping[key]:
                simplified_label= list(self.simplified_label_mapping.keys()).index(key)
        return np.array(int(simplified_label))
    
    def read_audio_from_npy(self, path):
        path = path[0]
        audio_data = np.load(f'{path}', allow_pickle=True)
        return np.array([audio_data], np.float32)
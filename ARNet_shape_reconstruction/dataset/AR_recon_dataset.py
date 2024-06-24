import re
import os
import sys
import random
import torch
import pickle
import numpy as np
sys.path.append('.')
import open3d as o3d
from operator import add 
import pytorch_lightning as pl

class AR_recon_dataset_s2r(pl.LightningDataModule):
    def __init__(self, split,epoch,params):
        assert split in ['train', 'valid', 'test']
        self.params = params
        self.epoch=epoch
        self.split = split
        self.gt_paths,self.input_paths = self._load_data()

    def __getitem__(self, index):
        output_path = self.gt_paths[index][0]
        input_path = self.input_paths[index][0]
        object_name=output_path.replace('.npy','')
        object_name=object_name.replace('data/ARDataset/ground_truth_5000_correct_unit/','')
        idx = re.findall("-\d\d?", object_name)
        if len(idx)!=0:
            object_name = object_name.replace(idx[0], '')
        if self.gt_paths[index][1]=='real':
            input_pc = self.read_real_contact_point_from_npy(input_path)
            output_pc = self.read_real_point_cloud_from_npy(output_path)
        else:
            input_pc = self.read_syn_contact_point_from_npy(input_path)
            output_pc = self.read_syn_point_cloud_from_npy(output_path)    
        return torch.from_numpy(input_pc), torch.from_numpy(output_pc)

    def __len__(self):
        return len(self.gt_paths)

    def _load_data(self):
        syn_gt_path = f'dataset/{self.params.exp_name}/syn_train_gt.npy'
        syn_input_path = f'dataset/{self.params.exp_name}/syn_train_tapping.npy'
        syn_gt_list=np.load(syn_gt_path,allow_pickle=True)
        syn_input_list=np.load(syn_input_path,allow_pickle=True)
        
        for i in range(len(syn_gt_list)):
            syn_gt_list[i]=[syn_gt_list[i],'syn']
            syn_input_list[i]=[syn_input_list[i],'syn']

        real_gt_path = f'dataset/{self.params.exp_name}/real_{self.split}_gt.npy'
        real_input_path = f'dataset/{self.params.exp_name}/real_{self.split}_tapping.npy'
        real_gt_list=np.load(real_gt_path,allow_pickle=True)
        real_input_list=np.load(real_input_path,allow_pickle=True)
        for i in range(len(real_gt_list)):
            real_gt_list[i]=[real_gt_list[i],'real']
            real_input_list[i]=[real_input_list[i],'real']
        if self.split == 'test' or self.split == 'valid':
            gt_path=real_gt_list
            input_path=real_input_list
        else:
            if self.epoch<100:
                syn_gt_list=syn_gt_list
                syn_input_list=syn_input_list
                real_gt_list=[]
                real_input_list=[]
            if  self.epoch>=100 and self.epoch<200:
                new_syn_input_list=random.sample(list(syn_input_list),int(0.9*len(syn_input_list)))
                new_syn_gt_list=[]
                for i in new_syn_input_list:
                    index=list(syn_input_list).index(i)
                    new_syn_gt_list.append(syn_gt_list[index])
                syn_gt_list=new_syn_gt_list
                syn_input_list=new_syn_input_list

                new_real_input_list=random.sample(list(real_input_list),int(0.1*len(real_input_list)))
                new_real_gt_list=[]
                for i in new_real_input_list:
                    index=list(real_input_list).index(i)
                    new_real_gt_list.append(real_gt_list[index])
                real_gt_list=new_real_gt_list
                real_input_list=new_real_input_list

            if  self.epoch>=200 and self.epoch<300:
                new_syn_input_list=random.sample(list(syn_input_list),int(0.8*len(syn_input_list)))
                new_syn_gt_list=[]
                for i in new_syn_input_list:
                    index=list(syn_input_list).index(i)
                    new_syn_gt_list.append(syn_gt_list[index])
                syn_gt_list=new_syn_gt_list
                syn_input_list=new_syn_input_list

                new_real_input_list=random.sample(list(real_input_list),int(0.2*len(real_input_list)))
                new_real_gt_list=[]
                for i in new_real_input_list:
                    index=list(real_input_list).index(i)
                    new_real_gt_list.append(real_gt_list[index])
                real_gt_list=new_real_gt_list
                real_input_list=new_real_input_list

            if  self.epoch>=300 and self.epoch<400:
                new_syn_input_list=random.sample(list(syn_input_list),int(0.6*len(syn_input_list)))
                new_syn_gt_list=[]
                for i in new_syn_input_list:
                    index=list(syn_input_list).index(i)
                    new_syn_gt_list.append(syn_gt_list[index])
                syn_gt_list=new_syn_gt_list
                syn_input_list=new_syn_input_list

                new_real_input_list=random.sample(list(real_input_list),int(0.4*len(real_input_list)))
                new_real_gt_list=[]
                for i in new_real_input_list:
                    index=list(real_input_list).index(i)
                    new_real_gt_list.append(real_gt_list[index])
                real_gt_list=new_real_gt_list
                real_input_list=new_real_input_list
            if  self.epoch>=400 and self.epoch<500:
                new_syn_input_list=random.sample(list(syn_input_list),int(0.4*len(syn_input_list)))
                new_syn_gt_list=[]
                for i in new_syn_input_list:
                    index=list(syn_input_list).index(i)
                    new_syn_gt_list.append(syn_gt_list[index])
                syn_gt_list=new_syn_gt_list
                syn_input_list=new_syn_input_list

                new_real_input_list=random.sample(list(real_input_list),int(0.6*len(real_input_list)))
                new_real_gt_list=[]
                for i in new_real_input_list:
                    index=list(real_input_list).index(i)
                    new_real_gt_list.append(real_gt_list[index])
                real_gt_list=new_real_gt_list
                real_input_list=new_real_input_list

            if  self.epoch>=500 and self.epoch<600:
                new_syn_input_list=random.sample(list(syn_input_list),int(0.2*len(syn_input_list)))
                new_syn_gt_list=[]
                for i in new_syn_input_list:
                    index=list(syn_input_list).index(i)
                    new_syn_gt_list.append(syn_gt_list[index])
                syn_gt_list=new_syn_gt_list
                syn_input_list=new_syn_input_list

                new_real_input_list=random.sample(list(real_input_list),int(0.8*len(real_input_list)))
                new_real_gt_list=[]
                for i in new_real_input_list:
                    index=list(real_input_list).index(i)
                    new_real_gt_list.append(real_gt_list[index])
                real_gt_list=new_real_gt_list
                real_input_list=new_real_input_list
            if  self.epoch>=600 and self.epoch<700:
                new_syn_input_list=random.sample(list(syn_input_list),int(0.1*len(syn_input_list)))
                new_syn_gt_list=[]
                for i in new_syn_input_list:
                    index=list(syn_input_list).index(i)
                    new_syn_gt_list.append(syn_gt_list[index])
                syn_gt_list=new_syn_gt_list
                syn_input_list=new_syn_input_list

                new_real_input_list=random.sample(list(real_input_list),int(0.9*len(real_input_list)))
                new_real_gt_list=[]
                for i in new_real_input_list:
                    index=list(real_input_list).index(i)
                    new_real_gt_list.append(real_gt_list[index])
                real_gt_list=new_real_gt_list
                real_input_list=new_real_input_list
            if  self.epoch>=700 and self.epoch<800:
                new_syn_input_list=random.sample(list(syn_input_list),int(0.05*len(syn_input_list)))
                new_syn_gt_list=[]
                for i in new_syn_input_list:
                    index=list(syn_input_list).index(i)
                    new_syn_gt_list.append(syn_gt_list[index])
                syn_gt_list=new_syn_gt_list
                syn_input_list=new_syn_input_list

                new_real_input_list=random.sample(list(real_input_list),int(0.95*len(real_input_list)))
                new_real_gt_list=[]
                for i in new_real_input_list:
                    index=list(real_input_list).index(i)
                    new_real_gt_list.append(real_gt_list[index])
                real_gt_list=new_real_gt_list
                real_input_list=new_real_input_list

            if  self.epoch>=800:
                syn_gt_list=[]
                syn_input_list=[]
                real_gt_list=real_gt_list
                real_input_list=real_input_list

            gt_path=[]
            input_path=[]
            for i in syn_gt_list:
                gt_path.append(i)
            for i in real_gt_list:
                gt_path.append(i)
            for i in syn_input_list:
                input_path.append(i)
            for i in real_input_list:
                input_path.append(i)

        print(len(gt_path),len(input_path))
        return gt_path,input_path
    
    def read_syn_point_cloud_from_npy(self, path):
        data = np.load(f'{path}', allow_pickle='TRUE')
        points = []
        for i in data:
            points.append(i[0:3])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=float))
        return np.array(pcd.points, np.float32)
    
    def read_syn_contact_point_from_npy(self, path):
        data = np.load(f'{path}', allow_pickle='TRUE')
        points = []
        for i in data:
            points.append(i[0:3])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        return np.array(pcd.points, np.float32)

    def read_real_point_cloud_from_npy(self, path):
        data = np.load(f'{path}', allow_pickle='TRUE')
        points = []
        for i in data:
            points.append(i[0:3])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=float))
        return np.array(pcd.points, np.float32)
    
    def read_real_contact_point_from_npy(self, path):
        data = np.load(f'{path}', allow_pickle='TRUE')
        points = []
        for i in data:
            points.append(i[0:3])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        return np.array(pcd.points, np.float32)

    def read_real_point_cloud_from_npy_and_transform_to_origin(self, path,object_name):
        transform_path = f'dataset/transformation/{object_name}_transformation_matrix.pkl'
        with open(transform_path, 'rb') as f:
            T = pickle.load(f)[0]
        invT=np.linalg.inv(T)
        R=invT[0:3,0:3]
        Trans=T[0:3,3]
        data = np.load(f'{path}', allow_pickle='TRUE')
        points = []
        for i in data:
            if sum(i[0:3])==0:
                points.append(i[0:3])
            else:
                transformed_points=list(map(add, i[0:3], -Trans)) 
                points.append(transformed_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=float))
        pcd.rotate(R,center=[0,0,0])
        return np.array(pcd.points, np.float32)
    
    def read_contact_point_from_npy(self, path):
        data = np.load(f'{path}', allow_pickle='TRUE')
        points = []
        for i in data:
            points.append(i[0:3])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=float))
        return np.array(pcd.points, np.float32)
    
    def read_real_contact_point_from_npy_and_transform_to_origin(self, path,object_name):
        transform_path = f'dataset/transformation/{object_name}_transformation_matrix.pkl'
        with open(transform_path, 'rb') as f:
            T = pickle.load(f)[0]
        invT=np.linalg.inv(T)
        R=invT[0:3,0:3]
        Trans=T[0:3,3]
        data = np.load(f'{path}', allow_pickle='TRUE')
        points = []
        for i in data:
            if sum(i[0:3])==0:
                points.append(i[0:3])
            else:
                transformed_points=list(map(add, i[0:3], -Trans)) 
                points.append(transformed_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=float))
        pcd.rotate(R,center=[0,0,0])

        return np.array(pcd.points, np.float32)
    
    def random_sample(self, pc, n):
        idx = np.random.permutation(pc.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
        return pc[idx[:n]]

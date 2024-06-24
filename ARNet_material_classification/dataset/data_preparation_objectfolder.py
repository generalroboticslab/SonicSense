import os
import re
import sys
import json
import yaml
import random
import librosa
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pytorch_lightning as pl
sys.path.append('.')

class data_preparation(object):
    def __init__(self,params):
        pl.seed_everything(1)
        self.split = None
        self.exp=params.exp_name
        self.current_label_list=params.label_mapping
        for folder in ['images','results',f'results/{params.exp_name}',f'results/{params.exp_name}/test',f'results/{params.exp_name}/valid']:
            try:
                os.makedirs(folder)
            except:
                pass

        with open('dataset/material_simple_categories.json') as f:
            self.simplified_label_mapping = json.load(f)

    def get_even_dataset(self):
        #0,3,4,6,8 corresponding to orignial material label
        all_object_list=[]
        for key in params.objectfolder_label_mapping:
            for i in params.objectfolder_label_mapping[key]:
                all_object_list.append(i)
        print('total num of objects',len(all_object_list))

        audio_list = []
        label_list = []

        for key in params.objectfolder_label_mapping:
            for i in params.objectfolder_label_mapping[key]:
                for file_name in os.listdir(f'objectfolder_data/extracted_audio_npy/{i}/audio'):

                    audio_list.append(f'objectfolder_data/extracted_audio_npy/{i}/audio/{file_name}')
                    label_list.append(key)
        print(len(audio_list),len(label_list))
        labels_statistic = [label_list.count(0),label_list.count(1),label_list.count(2),label_list.count(3),label_list.count(4)]
        max_number = np.max(labels_statistic)
        print('label statistic:', labels_statistic)
        num_data_for_validation=[int(0.1*i) for i in labels_statistic]

        audio_list_splitbyclass = [[] for _ in range(5)]
        label_list_splitbyclass = [[] for _ in range(5)]

        for idx, label in enumerate(label_list):
            audio_list_splitbyclass[label].append(audio_list[idx])
            label_list_splitbyclass[label].append(label)
        valid_audio_list=[[] for _ in range(5)]
        valid_label_list=[[] for _ in range(5)]

        for idx,i in enumerate(audio_list_splitbyclass):
            valid_audio_list[idx] =  i[0:num_data_for_validation[idx]]
            valid_label_list[idx] = label_list_splitbyclass[idx][0:num_data_for_validation[idx]]
        train_audio_list=[[] for _ in range(5)]
        train_label_list=[[] for _ in range(5)]

        for idx,i in enumerate(audio_list_splitbyclass):
            train_audio_list[idx] =  i[num_data_for_validation[idx]::]
            train_label_list[idx] = label_list_splitbyclass[idx][num_data_for_validation[idx]::]


        max_valid = np.max([len(valid_audio_list[0]),len(valid_audio_list[1]),len(valid_audio_list[2]),len(valid_audio_list[3]),len(valid_audio_list[4])])
        max_train = np.max([len(train_audio_list[0]),len(train_audio_list[1]),len(train_audio_list[2]),len(train_audio_list[3]),len(train_audio_list[4])])

        
        for idx,i in enumerate(valid_audio_list):
            # print(i)
            while len(i) != max_valid:
                audio_file = np.random.choice(i)
                index=audio_list.index(audio_file)
                valid_audio_list[idx].append(audio_file)
                valid_label_list[idx].append(label_list[index])

    
        for idx,i in enumerate(train_audio_list):
            # print(i)
            while len(i) != max_train:
                audio_file = np.random.choice(i)
                index=audio_list.index(audio_file)
                train_audio_list[idx].append(audio_file)
                train_label_list[idx].append(label_list[index])

        final_valid_list_label = []
        final_train_list_label = []
        final_valid_list_audio = []
        final_train_list_audio = []

        for idx,i in enumerate(valid_label_list):
            for idx_2,j in enumerate(i):
                final_valid_list_label.append(j)
                final_valid_list_audio.append(valid_audio_list[idx][idx_2])
        for idx,i in enumerate(train_label_list):
            for idx_2,j in enumerate(i):
                final_train_list_label.append(j)
                final_train_list_audio.append(train_audio_list[idx][idx_2])

        print(len(final_train_list_audio),len(final_valid_list_audio))
        for i in final_valid_list_audio:
            if i in final_train_list_audio:

                print('error========',i)

        np.save('data/ARdataset/object_folder/train_audio_list',np.array(final_train_list_audio, dtype=object))
        np.save('data/ARdataset/object_folder/train_label_list',np.array(final_train_list_label, dtype=object))
        np.save('data/ARdataset/object_folder/valid_audio_list',np.array(final_valid_list_audio, dtype=object))
        np.save('data/ARdataset/object_folder/valid_label_list',np.array(final_valid_list_label, dtype=object))


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
    
    def read_label_from_npy(self, path):
        path, idx = path
        if 'objectfolder' in path:
            label=1
        else:
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
        # path = path[0]
        audio_data = np.load(f'{path}', allow_pickle=True)
        return np.array([audio_data], np.float32)

    def show_guassian_noise(self):
        audio_paths=np.load(f'data/ARdataset//{self.split}_audio_list.npy', allow_pickle=True)
        for idx,i in enumerate(audio_paths):
            audio_path=i[0].replace('.wav','.npy')
            audio = self.read_audio_from_npy([audio_path])
            librosa.display.specshow(audio[0], x_axis='time', y_axis='mel',sr=44100,cmap='inferno')
            plt.colorbar(cmap='inferno',format='%+2.f')
            plt.savefig(f'images/origianl_{idx}',dpi=300)
            plt.close()
            gaussian = np.random.normal(-27.56679, 2, (64, 64))
            gaussian = np.array([gaussian], np.float32)
            audio = audio + gaussian
            librosa.display.specshow(audio[0], x_axis='time', y_axis='mel',sr=44100,cmap='inferno')
            plt.colorbar(cmap='inferno',format='%+2.f')
            plt.savefig(f'images/{idx}',dpi=300)
            plt.close()
            input()

if __name__ == '__main__':

    def load_config(filepath):
        with open(filepath, 'r') as stream:
            try:
                trainer_params = yaml.safe_load(stream)
                return trainer_params
            except yaml.YAMLError as exc:
                print(exc)
    from munch import munchify
    params = load_config(filepath='configs/config.yaml')
    params = munchify(params)
    print(params.objectfolder_label_mapping[0][0])
    data=data_preparation(params)
    data.get_even_dataset()


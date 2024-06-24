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

        label_data_path_origin = 'data/ARdataset/material_label'
        audio_data_path_origin = 'data/ARdataset/audio'
        for self.split in ['train','valid','test']:
            list_of_object_name=np.load(f'data/ARdataset/{self.exp}/{self.split}_object_list.npy', allow_pickle=True)
            #collect all the data path in one list
            material_label_paths = []
            audio_paths = []
            for path in list_of_object_name:
                # check if current path is a file
                if os.path.isfile(os.path.join(label_data_path_origin, path)):
                    audio_file_name = path.replace('.npy', '')
                    list_of_file=os.listdir(f'{audio_data_path_origin}/{audio_file_name}')
                    try:
                        list_of_file.remove('.DS_Store')
                    except:
                        pass
                    for i in range(len(list_of_file)):
                        filename=list_of_file[i]
                        if 'augmented' in filename:
                            filename=filename.replace('augmented1_','')
                            filename=filename.replace('augmented2_','')

                        data_idx=self.get_index_from_audiofilename(filename)
                        material_label_paths.append([f'{label_data_path_origin}/{path}', data_idx])
                        audio_paths.append([f'{audio_data_path_origin}/{audio_file_name}/{list_of_file[i]}'])

            #store file name according to index 
            labels=[0 for _ in range(43)]
            material_path=[[] for _ in range(43)]
            audio_path=[[] for _ in range(43)]
            num_43=0
            print(len(material_label_paths))
            non_43_material_label_path=[]
            non_43_audio_path=[]
            for i in range(len(material_label_paths)):
                label = self.read_label_from_npy(material_label_paths[i])
                if label==43:
                    num_43+=1
                else:
                    material_path[label].append(material_label_paths[i])
                    audio_path[label].append(audio_paths[i])
                    labels[label]+=1
                    non_43_material_label_path.append(material_label_paths[i])
                    non_43_audio_path.append(audio_paths[i])
            print('original material distribution:',labels)
            print('number of 43:',num_43)
            idx=[]
            for i in self.current_label_list:
                idx.append(i)
            k=len(self.current_label_list)
            print('material idx =', idx)
            num_data=[0 for _ in range(k)]
            material_labels_for_learning=[[] for _ in range(k)]
            audio_path_for_learning=[[] for _ in range(k)]
            for i in range(len(non_43_material_label_path)):
                label = self.read_simplified_label_from_npy(non_43_material_label_path[i])
                if label in idx:
                    material_labels_for_learning[int(np.where(idx == label)[0])].append(non_43_material_label_path[i])
                    audio_path_for_learning[int(np.where(idx == label)[0])].append(non_43_audio_path[i])
                    num_data[int(np.where(idx == label)[0])]+=1
            print(num_data)
            #split dataset
            #random shuffle data to randomly split data in testing, validation and training set
            #save original data list
            list_of_test_data_label=[]
            list_of_test_data_audio=[]
            for i in range(k):
                for idx,j in enumerate(material_labels_for_learning[i]):
                    list_of_test_data_label.append(j)
                    list_of_test_data_audio.append(audio_path_for_learning[i][idx])

            # save data in npy file
            name=[f'{self.split}_label_list',f'{self.split}_audio_list']
            data=[list_of_test_data_label
                    ,list_of_test_data_audio]
            for i in range(len(name)):
                print(name[i])
                dnpy = np.array(data[i], dtype=object)
                np.save(f'data/ARdataset/{self.exp}/original_{name[i]}', dnpy)

            if self.split=='train' or self.split=='valid' or self.split=='test':
                #duplicate data to even the dataset
                if self.split == 'train':
                    test_maximum=np.max(num_data)
                else:
                    test_maximum=np.max(num_data)
                for i in range(k):
                    while num_data[i]!=test_maximum and num_data[i]!=0 :
                        path=audio_path_for_learning[i][random.randint(0, len(audio_path_for_learning[i])-1)]

                        audio_path_for_learning[i].append(path)
                        num_data[i]+=1
                        
                print(f'{self.split} data statistc:', num_data)

                # recreate list of data for finding corresponding audio data
                list_of_test_data_label=[]
                list_of_test_data_audio=[]

                for i in range(k):
                    for j in range(len(audio_path_for_learning[i])):
                        path=audio_path_for_learning[i][j]
                        index=audio_paths.index(path)
                        label_path=material_label_paths[index]

                        label = int(self.read_simplified_label_from_npy(label_path))
                        if label not in self.current_label_list:
                            pass
                        else:
                            list_of_test_data_label.append(label_path)
                            list_of_test_data_audio.append(path)
                print(len(list_of_test_data_audio),len(list_of_test_data_label))
            
                # save data in npy file
                name=[f'{self.split}_label_list',f'{self.split}_audio_list']
                data=[list_of_test_data_label
                        ,list_of_test_data_audio]
                for i in range(len(name)):
                    dnpy = np.array(data[i], dtype=object)
                    np.save(f'data/ARdataset/{self.exp}/{name[i]}', dnpy)

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
        path = path[0]
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
    data=data_preparation(params)
    data.get_even_dataset()


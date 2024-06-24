import os
import re
import sys
import yaml
import random
import numpy as np
from munch import munchify

class data_preparation(object):
    def __init__(self):
        #build necessary folder
        for folder in ['image','reconstruction_results']:
            for data_split in ['data_split_1','data_split_2','data_split_3']:
                for split in ['train','valid','test']:
                    try:
                        os.makedirs(f'{folder}/{data_split}/{split}')
                    except:
                        pass

    def prepare_real_dataset(self, data_split):
        filepath = f'configs/config_{data_split}.yaml'
        with open(filepath, 'r') as stream:
            args = yaml.safe_load(stream)
            args = munchify(args)
        #train
        all_tapping_data_list=[]
        for object_name in os.listdir(f'data/ARDataset/{args.exp_name}/augmented_tapping_points_train'):
            for id in os.listdir(f'data/ARDataset/{args.exp_name}/augmented_tapping_points_train/{object_name}'):
                all_tapping_data_list.append(f'data/ARDataset/{args.exp_name}/augmented_tapping_points_train/{object_name}/{id}')
        print(len(all_tapping_data_list))

        random.shuffle(all_tapping_data_list)
        all_gt_data_list=[]
        num=0
        for i in all_tapping_data_list:
            list_of_gt=os.listdir('data/ARDataset/ground_truth_5000_correct_unit')
            get=False
            for gt_name in list_of_gt:
                gt_name=gt_name.replace('.npy','')
                if '_'+gt_name in i or '/'+gt_name in i:
                    all_gt_data_list.append(f'data/ARDataset/ground_truth_5000_correct_unit/{gt_name}.npy')
                    get=True
            if get == False:
                print(i)

        print(len(all_gt_data_list))
        train_gt_npy=np.array(all_gt_data_list,dtype=object)
        train_tapping_npy=np.array(all_tapping_data_list,dtype=object)

        np.save(f'dataset/{args.exp_name}/real_train_gt.npy',train_gt_npy)
        np.save(f'dataset/{args.exp_name}/real_train_tapping.npy',train_tapping_npy)

        # validation
        all_tapping_data_list=[]
        object_list=[]
        for object_name in os.listdir(f'data/ARDataset/{args.exp_name}/tapping_points_valid'):
            for id in os.listdir(f'data/ARDataset/{args.exp_name}/tapping_points_valid/{object_name}'):
                object_list.append(object_name)
                all_tapping_data_list.append(f'data/ARDataset/{args.exp_name}/tapping_points_valid/{object_name}/{id}')

        random.shuffle(all_tapping_data_list)
        all_gt_data_list=[]
        for i in all_tapping_data_list:
            list_of_gt=os.listdir('data/ARDataset/ground_truth_5000_correct_unit')
            for gt_name in list_of_gt:
                gt_name=gt_name.replace('.npy','')
                if gt_name in i:
                    all_gt_data_list.append(f'data/ARDataset/ground_truth_5000_correct_unit/{gt_name}.npy')
        valid_gt_npy=np.array(all_gt_data_list,dtype=object)
        valid_tapping_npy=np.array(all_tapping_data_list,dtype=object)
        valid_object_npy=np.array(object_list,dtype=object)

        print(len(valid_tapping_npy))
        np.save(f'dataset/{args.exp_name}/real_valid_gt.npy',valid_gt_npy)
        np.save(f'dataset/{args.exp_name}/real_valid_tapping.npy',valid_tapping_npy)
        np.save(f'dataset/{args.exp_name}/valid_object_list.npy',valid_object_npy)

        #testing
        all_tapping_data_list=[]
        object_list=[]
        for object_name in os.listdir(f'data/ARDataset/{args.exp_name}/tapping_points_test'):
            for id in os.listdir(f'data/ARDataset/{args.exp_name}/tapping_points_test/{object_name}'):
                object_list.append(object_name)
                all_tapping_data_list.append(f'data/ARDataset/{args.exp_name}/tapping_points_test/{object_name}/{id}')

        random.shuffle(all_tapping_data_list)
        all_gt_data_list=[]
        for i in all_tapping_data_list:
            list_of_gt=os.listdir('data/ARDataset/ground_truth_5000_correct_unit')
            for gt_name in list_of_gt:
                gt_name=gt_name.replace('.npy','')
                if gt_name in i:
                    all_gt_data_list.append(f'data/ARDataset/ground_truth_5000_correct_unit/{gt_name}.npy')
                    
        valid_gt_npy=np.array(all_gt_data_list,dtype=object)
        valid_tapping_npy=np.array(all_tapping_data_list,dtype=object)
        valid_object_npy=np.array(object_list,dtype=object)

        print(len(valid_tapping_npy))
        np.save(f'dataset/{args.exp_name}/real_test_gt.npy',valid_gt_npy)
        np.save(f'dataset/{args.exp_name}/real_test_tapping.npy',valid_tapping_npy)
        np.save(f'dataset/{args.exp_name}/test_object_list.npy',valid_object_npy)

        print('real dataset all set')

    def prepare_synthetic_dataset(self,data_split):
        filepath = f'configs/config_{data_split}.yaml'
        with open(filepath, 'r') as stream:
            args = yaml.safe_load(stream)
            args = munchify(args)

        all_tapping_data_list=[]
        for object_name in os.listdir('data/ARDataset_synthetic_data/augmented_synthetic_tapping_dataset'):
            for id in os.listdir(f'data/ARDataset_synthetic_data/augmented_synthetic_tapping_dataset/{object_name}'):
                all_tapping_data_list.append(f'data/ARDataset_synthetic_data/augmented_synthetic_tapping_dataset/{object_name}/{id}')
        print(len(all_tapping_data_list))

        random.shuffle(all_tapping_data_list)
        all_gt_data_list=[]
        for i in all_tapping_data_list:
            idx = re.findall("-\d\d?", i)
            if len(idx)!=0:
                i = i.replace(idx[0], '')
            idx = re.findall("/\d\d?_", i)
            if len(idx)!=0:
                i = i.replace(idx[0], '/')
            all_gt_data_list.append(i.replace('augmented_synthetic_tapping_dataset','syn_ground_truth_pcd'))

        train_gt_npy=np.array(all_gt_data_list,dtype=object)
        train_tapping_npy=np.array(all_tapping_data_list,dtype=object)

        np.save(f'dataset/{args.exp_name}/syn_train_gt.npy',train_gt_npy)
        np.save(f'dataset/{args.exp_name}/syn_train_tapping.npy',train_tapping_npy)

        train_gt_list=np.load(f'dataset/{args.exp_name}/syn_train_gt.npy',allow_pickle=True)
        print(len(train_gt_list))
        shape_list=['/bottle/','/cube/','/can/','/cylinder/','/cup/','/hammer/','/mug/','/cone/','/quadrangular_pyramid/','/triangular_prism/','/triangular_pyramid/','/wine_glass/']
        num_list=[0 for _ in range(len(shape_list))]
        for idx,i in enumerate(shape_list):

            for j in train_gt_list:
                if i in j:
                    num_list[idx]+=1
        print(num_list)
        print(sum(num_list))
        print('synthetic dataset all set')


if __name__ == '__main__':
    data_pre = data_preparation()
    choice_of_config = sys.argv[1]
    data_pre.prepare_real_dataset(choice_of_config)
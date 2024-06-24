import os
import math
import random
import numpy as np

#random baseline
for split in ['data_split_3']:
    test_gt_list = []
    train_gt_list = []
    for shape in os.listdir(f'data/ARDataset/{split}/tapping_points_test'):
        for object_name in os.listdir(f'data/ARDataset/{split}/tapping_points_test/{shape}'):
            test_gt_list.append(f'{object_name}')
    for shape in os.listdir(f'data/ARDataset/{split}/tapping_points_train'):
        for object_name in os.listdir(f'data/ARDataset/{split}/tapping_points_train/{shape}'):
            train_gt_list.append(f'{object_name}')
    for i in range(len(test_gt_list)):
        test_gt_list[i] = f'data/ARDataset/ground_truth_5000_correct_unit/'+test_gt_list[i]
    for i in range(len(train_gt_list)):
        train_gt_list[i] = f'data/ARDataset/ground_truth_5000_correct_unit/'+train_gt_list[i]
    for shape in os.listdir(f'data/ARDataset_synthetic_data/syn_ground_truth_pcd'):
        for object_name in os.listdir(f'data/ARDataset_synthetic_data/syn_ground_truth_pcd/{shape}'):
            train_gt_list.append(f'data/ARDataset_synthetic_data/syn_ground_truth_pcd/{shape}/{object_name}')
    loss = []
    for object_name in test_gt_list:
        test_gt = np.load(object_name, allow_pickle = True)
        train_object_name = random.choice(train_gt_list)
        train_gt = np.load(train_object_name,allow_pickle = True)
        print(split, object_name,train_object_name)

        distanct_1=[]
        for i in test_gt:
            min_dis=float('inf')
            for j in train_gt:
                distance = distance=math.dist(i,j)
                if distance <min_dis:
                    min_dis=distance
            distanct_1.append(min_dis)
        loss_1=sum(distanct_1)/len(test_gt)
    
        distanct_2=[]
        for i in train_gt:
            min_dis=float('inf')
            for j in test_gt:
                distance = distance=math.dist(i,j)
                if distance <min_dis:
                    min_dis=distance
            distanct_2.append(min_dis)
        loss_2=sum(distanct_2)/len(train_gt)
        loss.append(loss_1+loss_2)
        print('l1CD:',loss_1+loss_2)
    print('average l1CD:',np.mean(loss))

#nearest neighbor baseline
for split in ['data_split_3']:
    test_input_list = np.load(f'dataset/{split}/real_test_tapping.npy',allow_pickle=True)
    test_gt_list = np.load(f'dataset/{split}/real_test_gt.npy',allow_pickle=True)
    real_train_input_list = np.load(f'dataset/{split}/real_train_tapping.npy',allow_pickle=True)
    syn_train_input_list = np.load(f'dataset/{split}/syn_train_tapping.npy',allow_pickle=True)

    train_input_list = []
    for i in real_train_input_list:
        train_input_list.append(i)
    for i in syn_train_input_list:
        train_input_list.append(i)
    real_train_gt_list = np.load(f'dataset/{split}/real_train_gt.npy',allow_pickle=True)
    syn_train_gt_list = np.load(f'dataset/{split}/syn_train_gt.npy',allow_pickle=True)
    print(len(train_input_list))
    train_gt_list = []
    for i in real_train_gt_list:
        train_gt_list.append(i)
    for i in syn_train_gt_list:
        train_gt_list.append(i)
    print(len(train_gt_list))

    for test_idx, object_name in enumerate(test_input_list):
        test_input = np.load(object_name, allow_pickle = True)
        test_loss=[]
        min_loss = float('inf')
        number = 0
        for train_idx, i in enumerate(train_input_list):
            number+=1
            print(number)
            train_input = np.load(i,allow_pickle = True)
            distanct_1=[]
            for i in test_input:
                min_dis=float('inf')
                for j in train_input:
                    distance = distance=math.dist(i,j)
                    if distance <min_dis:
                        min_dis=distance
                distanct_1.append(min_dis)
            loss_1=sum(distanct_1)/len(test_input)
            distanct_2=[]
            for i in train_input:
                min_dis=float('inf')
                for j in test_input:
                    distance = distance=math.dist(i,j)
                    if distance <min_dis:
                        min_dis=distance
                distanct_2.append(min_dis)
            loss_2=sum(distanct_2)/len(train_input)
            loss = loss_1 + loss_2
            if loss < min_loss:
                min_loss = loss
                chosen_training_object_idx = train_idx
                
        print(split, test_gt_list[test_idx],train_gt_list[chosen_training_object_idx])
        test_gt = np.load(test_gt_list[test_idx], allow_pickle = True)
        train_gt = np.load(train_gt_list[chosen_training_object_idx],allow_pickle = True)
        distanct_1=[]
        for i in test_gt:
            min_dis=float('inf')
            for j in train_gt:
                distance = distance=math.dist(i,j)
                if distance <min_dis:
                    min_dis=distance
            distanct_1.append(min_dis)
        loss_1=sum(distanct_1)/len(test_gt)
        distanct_2=[]
        for i in train_gt:
            min_dis=float('inf')
            for j in test_gt:
                distance = distance=math.dist(i,j)
                if distance <min_dis:
                    min_dis=distance
            distanct_2.append(min_dis)
        loss_2=sum(distanct_2)/len(train_gt)
        loss = loss_1 + loss_2
        print('l1CD:', loss)
        test_loss.append(loss)
    print('average l1CD:',np.mean(test_loss))

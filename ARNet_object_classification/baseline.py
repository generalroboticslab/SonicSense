import os
import math
import copy
import yaml
import pickle
import random
import statistics
import numpy as np
import pandas as pd
import seaborn as sns
from munch import munchify
from natsort import natsorted
from sewar.full_ref import mse
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassConfusionMatrix

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)
params = load_config(filepath='configs/config.yaml')
params = munchify(params)
def get_data_list(split):
    contact_points_paths = []
    audio_paths = []
    label = []
    for file_name in natsorted(os.listdir(f'./data/contact_points/{split}')):
        contact_points_paths.append(f'./data/contact_points/{split}/{file_name}')
        for label_name in params.label_mapping:
            if f'_{label_name}.npy' in file_name:
                label.append(params.label_mapping[label_name])
    for path in contact_points_paths:
        path = path.replace('contact_points','audio')
        audio_paths.append(path)
    return contact_points_paths, audio_paths, label

def read_audio_from_npy(path):
    # path = path[0]
    audio_data = np.load(f'{path}', allow_pickle=True)
    return np.array([audio_data], np.float32)

def save_confusion_matrix(y_hat,y):
    metric = MulticlassConfusionMatrix(num_classes=params.num_label)
    cm=metric(y_hat,y)
    confusion_matrix_computed = cm.detach().cpu().numpy().astype(int)
    uniformed_confusion_matrix=[]
    for idx,i in enumerate(confusion_matrix_computed):
        uniformed_confusion_matrix.append([val/sum(i) for val in i])
    final_acc_list=[]
    for idx in range(len(uniformed_confusion_matrix)):
        final_acc_list.append(uniformed_confusion_matrix[idx][idx])
    final_acc=sum(final_acc_list)/len(final_acc_list)
    # print('final acc = ',final_acc)
    df_cm = pd.DataFrame(uniformed_confusion_matrix,index=params.label_name,columns=params.label_name)
    plt.figure(figsize = (10,8))
    fig_ = sns.heatmap(df_cm, annot=True, cmap='Reds').get_figure()
    plt.xlabel('Predicted labels')
    plt.ylabel('True lables')
    plt.savefig(f'images/{params.exp_name}_{params.testing_split}_baseline_confusion_matrix', dpi=300)
    plt.close(fig_)

def get_average_f1_score(y_hat,y):
    metric = MulticlassConfusionMatrix(num_classes=params.num_label)
    cm=metric(y_hat,y)
    confusion_matrix_computed = cm.detach().cpu().numpy().astype(int)
    precision_list = []
    for idx, row in enumerate(confusion_matrix_computed):
        p = row[idx]/sum(row)
        precision_list.append(p)
    recall_list = []
    for idx in range(params.num_label):
        i_column=[]
        for row in confusion_matrix_computed:
            i_column.append(row[idx])
        r = confusion_matrix_computed[idx][idx]/sum(i_column)
        recall_list.append(r)
    f1_score_list = []
    for i in range(params.num_label):
        f1_score_list.append(statistics.harmonic_mean([precision_list[i],recall_list[i]]))
    return np.mean(f1_score_list)

#Radom baseline
_,_,train_label_list = get_data_list('train')
_,_,test_label_list = get_data_list('test')
correct = 0
for i in test_label_list:
    prediction = random.choice(train_label_list)
    if i == prediction:
        correct+=1
print('random baseline accuracy',correct/len(test_label_list))

#Nearest neighbor baseline
train_contact_points,train_audio_path,train_label_list = get_data_list('train')
test_contact_points,test_audio_path,test_label_list = get_data_list('test')
print(len(train_contact_points),len(train_audio_path),len(train_label_list))
print(len(test_contact_points),len(test_audio_path),len(test_label_list))
correct = 0
print(len(test_label_list)/4)
for test_idx in range(int(len(test_label_list)/2),int(3*len(test_label_list)/4)):
    test_audio = read_audio_from_npy(test_audio_path[test_idx])
    test_point_cloud = np.load(test_contact_points[test_idx], allow_pickle = True)
    min_loss=float('inf')
    train_index = 0
    print(test_idx)
    for train_idx in range(len(train_label_list)):
        train_audio = read_audio_from_npy(train_audio_path[train_idx])
        train_point_cloud = np.load(train_contact_points[train_idx], allow_pickle = True)
        audio_loss = mse(train_audio,test_audio)
        distanct_1=[]
        for i in test_point_cloud:
            min_dis=float('inf')
            for j in train_point_cloud:
                distance = distance=math.dist(i,j)
                if distance <min_dis:
                    min_dis=distance
            distanct_1.append(min_dis)
        loss_1=sum(distanct_1)/len(test_point_cloud)
        distanct_2=[]
        for i in train_point_cloud:
            min_dis=float('inf')
            for j in test_point_cloud:
                distance = distance=math.dist(i,j)
                if distance <min_dis:
                    min_dis=distance
            distanct_2.append(min_dis)
        loss_2=sum(distanct_2)/len(train_point_cloud)
        points_loss = loss_1 + loss_2
        total_loss = audio_loss+points_loss
        if total_loss<min_loss:
            min_loss = total_loss
            train_index = train_idx
    if int(test_label_list[test_idx]) == int(train_label_list[train_index]):
        correct += 1
    print('accuracy', correct/len(test_label_list))
print('nearest neighbor accuracy', correct/len(test_label_list))
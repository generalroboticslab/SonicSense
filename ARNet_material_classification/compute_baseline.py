import json
import yaml
import torch
import random
import statistics
import numpy as np
import pandas as pd
import seaborn as sns
from munch import munchify
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

def read_simplified_label_from_npy(path):
    path, idx = path
    if 'objectfolder' in path:
        label=1
    else:
        idx=idx[1]+(idx[0]-1)*4
        data = [np.load(f'{path}', allow_pickle=True)[idx]]
        label=data[0]
    for key in simplified_label_mapping:
        if label in simplified_label_mapping[key]:
            simplified_label= list(simplified_label_mapping.keys()).index(key)
    return np.array(int(simplified_label))

def read_audio_from_npy(path):
    path = path[0]
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

with open('dataset/material_simple_categories.json') as f:
    simplified_label_mapping = json.load(f)

for split in ['data_split_1', 'data_split_2','data_split_3']:
    test_audio_list = np.load(f'data/ARdataset/{split}/original_test_audio_list.npy', allow_pickle = True)
    train_audio_list = np.load(f'data/ARdataset/{split}/train_audio_list.npy', allow_pickle = True)
    test_label_list = np.load(f'data/ARdataset/{split}/original_test_label_list.npy', allow_pickle = True)
    train_label_list = np.load(f'data/ARdataset/{split}/train_label_list.npy', allow_pickle = True)

    print(len(test_audio_list))
    print(len(train_audio_list))
    print(len(test_label_list))
    print(len(train_label_list))

    #random baseline
    correct=0
    params = load_config(filepath='configs/config.yaml')
    params = munchify(params)
    current_label_list=params.label_mapping
    test_labels=[]
    train_labels=[]
    for test_label_path in test_label_list:
        random_train_label_path = random.choice(train_label_list)
        test_label = np.array(current_label_list[int(read_simplified_label_from_npy(test_label_path))])
        train_label = np.array(current_label_list[int(read_simplified_label_from_npy(random_train_label_path))])
        test_labels.append(test_label)
        train_labels.append(train_label)
    #get confusion matrix
    save_confusion_matrix(torch.from_numpy(np.array(train_labels)),torch.from_numpy(np.array(test_labels)))
    #get f1 score
    average_f1_score = get_average_f1_score(torch.from_numpy(np.array(train_labels)),torch.from_numpy(np.array(test_labels)))
    print(split, 'avg f1 score:', average_f1_score)

    #nearest neighbor baseline
    correct=0
    params = load_config(filepath='configs/config.yaml')
    params = munchify(params)
    current_label_list=params.label_mapping
    test_labels=[]
    train_labels=[]
    for test_idx, test_audio_path in enumerate(test_audio_list):
        test_audio = read_audio_from_npy(test_audio_path)
        min_dis=float('inf')
        for train_idx, train_audio_path in enumerate(train_audio_list):
            train_audio = read_audio_from_npy(train_audio_path)
            distance = mse(train_audio,test_audio)
            if distance<min_dis:
                train_label_idx = train_idx
                min_dis = distance
        test_label = np.array(current_label_list[int(read_simplified_label_from_npy(test_label_list[test_idx]))])
        train_label = np.array(current_label_list[int(read_simplified_label_from_npy(train_label_list[train_label_idx]))])

        test_labels.append(test_label)
        train_labels.append(train_label)
    #get confusion matrix
    save_confusion_matrix(torch.from_numpy(np.array(train_labels)),torch.from_numpy(np.array(test_labels)))
    #get f1 score
    average_f1_score = get_average_f1_score(torch.from_numpy(np.array(train_labels)),torch.from_numpy(np.array(test_labels)))
    print(split, 'avg f1 score:', average_f1_score)






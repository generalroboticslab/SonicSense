import os
import math
import copy
import yaml
import torch
import pickle
import natsort
import statistics
import numpy as np
import pandas as pd
import open3d as o3d
import seaborn as sns
from munch import munchify
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassConfusionMatrix

class get_overall_accuracy(object):
    def __init__(self,params):
        self.num_class = params.num_label
        self.params = params

    def calculate_accuracy(self,split,k_neighbor,num_loop,mim_occurence):
        filepath = 'configs/config.yaml'
        with open(filepath, 'r') as stream:
            params = yaml.safe_load(stream)
            params = munchify(params)

        correct_label_list = params.label_mapping
        d_swap = {v: k for k, v in correct_label_list.items()}
        correct_label_list = d_swap
        npy_files = natsort.natsorted((np.load(f'data/ARdataset/{params.exp_name}/{split}_object_list.npy', allow_pickle=True)))
        file_name = np.load(f'data/ARdataset/{params.exp_name}/original_{split}_label_list.npy', allow_pickle=True)
        data = np.load(f'results/{params.exp_name}/{split}/label_prediction_{split}.npy', allow_pickle=True)
        gtdata = np.load(f'results/{params.exp_name}/{split}/label_gt_{split}.npy', allow_pickle=True)
        list_accuracy = []
        correct_prediction = []
        y_hat=[]
        y=[]
        num_correct_prediction = [0 for _ in range(len(params.label_name))]
        total_num_prediction = [0 for _ in range(len(params.label_name))]
        for file in npy_files:
            object_name = file.replace('.npy', '')
            raw_contact_points = self.read_raw_contact_from_txt(
                os.path.join('data/ARdataset/contact_position', file.replace('.npy', '.txt')))
            object_name_prediction = []
            label_prediction = []
            # not all of the contact point have valid audio data included in training, so here we extract the contact points with valid material prediction
            contact_points_idx_prediction = []
            ground_truth_label = []
            for idx, i in enumerate(file_name):
                # print( object_name)
                if object_name in i[0]:

                    object_name_prediction.append(i)
                    contact_points_idx_prediction.append((i[1][0] - 1) * 4 + i[1][1])
                    label_prediction.append(
                        correct_label_list[int(np.where(data[idx] == np.max(data[idx]))[0])])
                    ground_truth_label.append(correct_label_list[int(gtdata[idx])])


            predicted_contact_points = []
            for idx, value in enumerate(contact_points_idx_prediction):
                predicted_contact_points.append(raw_contact_points[value])
            # get statistic of occurence of labels and filter out labels with low occurance
            label_prediction,maxlabel = self.filter_out_labels_with_low_occurence(k_neighbor,mim_occurence, label_prediction,predicted_contact_points)

            # assign labels according to voting with nearest neighbors
            for i in range(num_loop):
                voted_label_list = []
                for idx, value in enumerate(predicted_contact_points):
                    voted_lable = self.get_labels_of_k_neighbor_points(k_neighbor, value,maxlabel, label_prediction,predicted_contact_points)
                    voted_label_list.append(voted_lable)
                label_prediction = voted_label_list.copy()

            #save confusion matrix
            for idx in range(len(label_prediction)):
                y_hat.append(params.label_mapping[int(label_prediction[idx])])
                y.append(params.label_mapping[int(ground_truth_label[idx])])

            
            # check number of correct prediction
            for idx, value in enumerate(label_prediction):
                pred_index = params.label_mapping[value]
                gt_index = params.label_mapping[ground_truth_label[idx]]
                if value == ground_truth_label[idx]:
                    num_correct_prediction[pred_index] += 1
                total_num_prediction[gt_index]+=1

        #save confusion matrix
        if split == 'test':
            self.save_confusion_matrix(torch.from_numpy(np.array(y_hat)),torch.from_numpy(np.array(y)))

        acc=[]
        for idx, i in enumerate(num_correct_prediction):
            acc.append(num_correct_prediction[idx]/total_num_prediction[idx])
        # calculate overall prediction accuracy
        accuracy_balanced = sum(acc) / len(acc)

        accuracy = sum(num_correct_prediction) / sum(total_num_prediction)
        list_accuracy.append(accuracy)
        return accuracy,accuracy_balanced
    def save_confusion_matrix(self,y_hat,y):
        print(y_hat.size(),y.size())
        print(len(y_hat),len(y))
        # for i in range(len(y_hat)):
        #     print(y_hat[i],y[i])
        #     input()
        
        metric = MulticlassConfusionMatrix(num_classes=self.num_class)
        cm=metric(y_hat,y)
        confusion_matrix_computed = cm.detach().cpu().numpy().astype(int)

        uniformed_confusion_matrix=[]
        for idx,i in enumerate(confusion_matrix_computed):
            uniformed_confusion_matrix.append([val/sum(i) for val in i])
        final_acc_list=[]
        for idx in range(len(uniformed_confusion_matrix)):
            final_acc_list.append(uniformed_confusion_matrix[idx][idx])
        final_acc=sum(final_acc_list)/len(final_acc_list)
        print('final acc = ',final_acc)

        df_cm = pd.DataFrame(uniformed_confusion_matrix,index=self.params.label_name,columns=self.params.label_name)
        plt.figure(figsize = (10,8))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Reds').get_figure()
        plt.xlabel('Predicted labels')
        plt.ylabel('True lables')
        plt.savefig(f'images/{self.params.exp_name}_{self.params.testing_split}_refine_confusion_matrix', dpi=300)
        plt.close(fig_)

    def read_raw_contact_from_txt(self,path):
        # extract [x,y,z] data from {contact_points}.txt file
        data = []
        with open(f'{path}', "r") as f:
            for line in f:
                ls = line.strip().split()
                ls = [float(i) for i in ls]
                data.append(ls)
        points = []
        for i in range(len(data)):
            points.append(data[i][0:3])
        #turn unit into meter and apply transformation for aligning contact points with 3D point cloud model
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=float))
        pcd.scale(0.01, [0, 0, 0])
        transform_path = 'data/ARdataset/transformation_matrix/tapping_position_transformation_matrix.pkl'
        with open(transform_path, 'rb') as f:
            T = pickle.load(f)[0]
        pcd_t = copy.deepcopy(pcd).transform(T)
        transform_path = 'data/ARdataset/transformation_matrix/tapping_position_transformation_matrix_1.pkl'
        with open(transform_path, 'rb') as f:
            T = pickle.load(f)[0]
        pcd_t = copy.deepcopy(pcd_t).transform(T)
        return np.asarray(pcd_t.points)

    def get_labels_of_k_neighbor_points(self,k, value, maxlabel,label_list,predicted_contact_points):
        neighbors_labels = []
        neighbors_points = []
        contact_points_list = predicted_contact_points.copy()
        contact_points_list = [list(i) for i in contact_points_list]
        for i in range(k):
            min = float('inf')
            for idx, points in enumerate(contact_points_list):
                #compute Euclidean Distance of two points
                distance = math.dist(value, points)
                if distance < min:
                    min = distance
                    key = idx
                    closest_point = points
            neighbors_points.append(contact_points_list[key])
            contact_points_list.remove(closest_point)
        contact_points_list = predicted_contact_points.copy()
        contact_points_list = [list(i) for i in contact_points_list]

        for idx, points in enumerate(contact_points_list):
            for value in neighbors_points:
                if list(value) == list(points):
                    neighbors_labels.append(label_list[idx])

        if len(statistics.multimode(neighbors_labels)) != 1:
            voted_label = maxlabel
        else:
            voted_label = statistics.multimode(neighbors_labels)[0]
        return voted_label


    def filter_out_labels_with_low_occurence(self,k_neighbor,mim_occurence,list_of_label,predicted_contact_points):
        labels = np.unique(list_of_label)
        labels_statistic=[]
        for i in labels:
            labels_statistic.append(list_of_label.count(i))
        maxidx=labels_statistic.index(max(labels_statistic))
        maxlabel=labels[maxidx]
        for i in range(len(labels_statistic)):
            if labels_statistic[i]<mim_occurence:
                for idx,value in enumerate(list_of_label):
                    if value==labels[i]:
                        # list_of_label[idx]=maxlabel

                        list_of_label[idx] = self.get_labels_of_k_neighbor_points(k_neighbor, predicted_contact_points[idx],
                                                                             maxlabel, list_of_label,predicted_contact_points)
        return list_of_label,maxlabel
    def get_best_params_from_val(self):
        acc=[]
        parameters=[]
        for k in range(1,9):
            for loop in range(1,9):
                for mim_occurrence in [5,15,20,25,30,35]:
                    _, acc_balanced=self.calculate_accuracy('valid', k,loop,mim_occurrence)
                    acc.append(acc_balanced)
                    print('current bast valid acc:', max(acc))

                    parameters.append([k,loop,mim_occurrence])
        print('final best valid acc:', max(acc))
        print(parameters[acc.index(max(acc))])

        return parameters[acc.index(max(acc))], max(acc)
    
    def get_best_test_acc(self):
        parameters, _=self.get_best_params_from_val()
        # parameters = [1,1,1]
        _, acc_balanced=self.calculate_accuracy('test', parameters[0],parameters[1],parameters[2])
        print('best test acc:', acc_balanced)

        return acc_balanced
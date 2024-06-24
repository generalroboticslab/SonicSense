import os
import math
import numpy as np

split='data_split_1'
loss_list=[]
file_list=[]
for file_name in os.listdir(f'reconstruction_results/{split}'):
    reconstruction_results=np.load(f'reconstruction_results/{split}/{file_name}',allow_pickle=True)[0]
    gt=np.load(f'data/ARDataset/ground_truth_5000_correct_unit/{file_name}',allow_pickle=True)
    distanct_1=[]
    for i in reconstruction_results:
        min_dis=float('inf')
        for j in gt:
            distance = distance=math.dist(i,j)
            if distance <min_dis:
                min_dis=distance
        distanct_1.append(min_dis)
    loss_1=sum(distanct_1)/len(reconstruction_results)
    
    distanct_2=[]
    for i in gt:
        min_dis=float('inf')
        for j in reconstruction_results:
            distance = distance=math.dist(i,j)
            if distance <min_dis:
                min_dis=distance
        distanct_2.append(min_dis)
    loss_2=sum(distanct_2)/len(gt)

    loss_list.append(loss_1+loss_2)
    file_list.append(file_name)
    print(loss_list,file_list)

for i in range(len(file_list)):
    print(file_list[i],loss_list[i])

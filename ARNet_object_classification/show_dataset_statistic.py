from munch import munchify
import yaml
import numpy as np
import json

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

def _load_data():
    material_label_paths=np.load(f'data/ARdataset/{exp}/{split}_label_list.npy', allow_pickle=True)
    audio_paths=np.load(f'data/ARdataset/{exp}/{split}_audio_list.npy', allow_pickle=True)
    labels=[0]*9

    #put the label filepath and audio filepath in its own class and count the number
    for i in range(len(material_label_paths)):
        label = current_label_list[int(read_simplified_label_from_npy(material_label_paths[i]))]
        labels[label]+=1 
    print(labels)
    print(f'number of {split} data =',len(material_label_paths) )
    for i in range(len(labels)):
        print(f'{split}_label_{i} % =',labels[i]/len(material_label_paths))
    print(split,'==========================total data:',len(material_label_paths))


with open('dataset/material_simple_categories.json') as f:
        simplified_label_mapping = json.load(f)


# exp='data_split_3'
# split='train'
# params = load_config(filepath='configs/config.yaml')
# params = munchify(params)
# current_label_list=params.label_mapping
# _load_data()
val=[56.67,54.84,57.23]
test=[56.06,53.65,52.88]
re_val=[72.35,81.95,78.24]
re_test=[81.35,78.12,69.79]

print(np.mean(re_val),np.std(re_val))
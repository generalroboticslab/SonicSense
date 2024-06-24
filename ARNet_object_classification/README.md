# ARNet_object_classification

## Installation
Uses the same virtual environment with material classification task.
```
source venv_classification/bin/activate
```

## Data Preparation

Download the dataset from the Google Drive and unzip under the folder.
* [checkpoint](https://drive.google.com/file/d/1WhwyrGYhOD5wQRjcUC6rdofhGc2phrWV/view?usp=sharing)
* [data_split_1](https://drive.google.com/file/d/12inxcmrpCIB7jT10Ef4WZTaRR4rfkZeL/view?usp=drive_link)
* [data_split_2](https://drive.google.com/file/d/1DCXEYFg8_IasN45InJUjvL-ZtbgTlaXm/view?usp=drive_link)
* [data_split_3](https://drive.google.com/file/d/1q4okQzeyyNRZCQXIB0qBeSOifWCE-1Of/view?usp=drive_link)


## Training

Run the following command:
```
CUDA_VISIBLE_DEVICES=6 python main.py <choice of datasplit> <choice of input type>
```

## Evaluation
Choose your ckpt file and change it in test_ckpt_path in config.yaml, then run the following command:
```
CUDA_VISIBLE_DEVICES=6 python test.py <choice of datasplit> <choice of input type>
```
* choice of datasplit : data_split_1/data_split_2/data_split_3
* choice of input type : audio+points/audio/points
The confusion matrices and numerical results are saved under `images` folder.
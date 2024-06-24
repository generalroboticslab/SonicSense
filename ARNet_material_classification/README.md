# ARNet_material_classification

## Installation
The code has been tested on Ubuntu 20.04 with CUDA 12.0.
```
virtualenv -p /usr/bin/python venv_classification
source venv_classification/bin/activate
cd ARNet_material_classification
pip install -r requirements.txt
```

## Data Preparation

Download the dataset from the [Google Drive](https://drive.google.com/file/d/121ZZw-_Bd2QLxrFwHMaK20bsRWV-05OF/view?usp=drive_link) and unzip under the folder.

## About Configs
- `exp_name` - used to specify data split
- `ckpt_path` - used to specify saved model for testing
- `init_dataset` - if it is True, the program with randomly rebalance the dataset, which will create a different dataset. Set it to False to reproduce the results.

## Training

Run the following command for training:
```
CUDA_VISIBLE_DEVICES=6 python main.py data_split_<1-3>
```
During training, the training and validation dataset are balanced, so the best model is selected based on the best validation accuracy.
## Evaluation
Run the following command for testing:
```
CUDA_VISIBLE_DEVICES=6 python test.py data_split_<1-3>
```
During testing, the testing dataset is unbalanced, so F1 score is used to evaluate the performance and the parameters of refinement method is searched on unbalanced validation dataset.

To test the model performance of the model trained on objectfolder dataset, run the following command:
```
CUDA_VISIBLE_DEVICES=6 python test_object_folder.py object_folder
```
The confusion matrices will be saved under `images` folder, and numerical results are saved under `results` folder.



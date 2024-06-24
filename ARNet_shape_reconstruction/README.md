# ARNet_shape_reconstruction 


## Installation
The code has been tested on Ubuntu 20.04 with CUDA 12.0.
```
virtualenv -p /usr/bin/python venv_recon
source venv_recon/bin/activate
cd ARNet_shape_reconstruction
pip install -r requirements.txt
```
## Chamfer Distance Loss
Please find the chamferdist/chamfer.py file under venv_recon/lib, and replace the chamfer.py in [this](https://github.com/krrish94/chamferdist?tab=readme-ov-file) python library to our chamfer.py file under the folder.

## Data Preparation

Download the dataset from the [Google Drive](https://drive.google.com/file/d/1kb4rS1cDhwCSQ4bM6IWvFHymnbocp2rh/view?usp=drive_link) and unzip under the folder.

## About Configs
- `exp_name` - used to specify data split
- `ckpt_path` - used to specify saved model for testing

## Training

Run the following command:
```
CUDA_VISIBLE_DEVICES=6 python main.py data_split_<1-3>
```

## Evaluation
Run the following command for evaluation of the trained models:
```
CUDA_VISIBLE_DEVICES=6 python test.py data_split_<1-3>
```
The visual reconstruction results are saved under `image` folder.The numerical reconstruction results will be saved in .npy file under `reconstruction_results` folder. 

import sys
import yaml
from datetime import date
from munch import munchify
import pytorch_lightning as pl
from models.ARnet_recon import ARnet_recon
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from visualization.visualization import plot_pcd_one_view
from dataset.data_preparation import data_preparation

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def main():
    choice_of_config = sys.argv[1]
    config_path = f'configs/config_{choice_of_config}.yaml'
    params = load_config(filepath=config_path)
    params = munchify(params)

    prepare_dataset = data_preparation()
    if params.init_dataset == True:
        prepare_dataset.prepare_real_dataset(choice_of_config)
        prepare_dataset.prepare_synthetic_dataset(choice_of_config)

    pl.seed_everything(1)
    today = date.today()
    today = str(today.strftime("%m/%d/%y")).replace('/','',2)
    logger = pl_loggers.TensorBoardLogger(save_dir=params.log_dir,name=params.exp_name)

    model = ARnet_recon(params=params)
    
    checkpoint_epoch_callback = ModelCheckpoint(dirpath=f'checkpoint',
                                                save_top_k=1,   
                                                every_n_epochs=100,
                                                filename=f'{today}/{params.exp_name}'+'-{epoch:02d}-{val_loss}')
    
    checkpoint_val_acc_callback = ModelCheckpoint(mode='min',
                                                  save_top_k=1,
                                                  monitor='val_loss',
                                                  dirpath=f'checkpoint',
                                                  filename=f'{today}/{params.exp_name}'+'-best_model-{epoch:02d}-{val_loss}')

    trainer=pl.Trainer(logger=logger,
                       accelerator="gpu",
                       deterministic=True,
                       log_every_n_steps=1,
                       max_epochs=params.epochs,
                       default_root_dir='checkpoint/',
                       callbacks=[checkpoint_epoch_callback,checkpoint_val_acc_callback],
                       reload_dataloaders_every_n_epochs=1)
    
    trainer.fit(model=model,ckpt_path=params.train_ckpt_path)    

if __name__ == '__main__':
    main()

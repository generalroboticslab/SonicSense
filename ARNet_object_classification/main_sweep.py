import yaml
import wandb
from torch import utils
from datetime import date
from munch import munchify
import pytorch_lightning as pl
from models.ARNet_object import ARNet_object
from pytorch_lightning.loggers import WandbLogger
from dataset.data_preparation import data_preparation
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset.AR_object_dataset import AR_object_dataset
from models.ARNet_object_larger import ARNet_object_larger
from models.ARNet_1024 import ARNet_object_1024

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def main():
    exp_name=params.exp_name
    sweep_config = {
        'method': 'random',
        'name': exp_name,
        'metric': {
            'goal': 'maximize', 
            'name': 'val_acc'
            },
        'parameters': {
            'batch_size': {'distribution': 'q_log_uniform_values',
            'q': 200,
            'min': 200,
            'max': 400,},
            'lr': {'distribution': 'uniform','max': 0.0001, 'min': 0.00001},
            'dropout': {
            'distribution': 'uniform','max': 0.4, 'min': 0.2
            },
        }
    }
    print(sweep_config,'=================')
    sweep_id = wandb.sweep(sweep_config, project="object_classification")
    wandb.agent(sweep_id, function=train, count=10)


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        config.lr = round(config.lr,6)
        config.dropout = round(config.dropout,4)
        pl.seed_everything(1)
        model = ARNet_object(params,config)

        wandb_logger = WandbLogger(log_model=True)

        train_dataset = AR_object_dataset( 'train',params)
        val_dataset = AR_object_dataset('valid',params)

        train_dataloader = utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=params.num_workers)
        val_dataloader = utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=params.num_workers)
        checkpoint_epoch_callback = ModelCheckpoint(dirpath=f'checkpoint',
                                                save_top_k=1,   
                                                every_n_epochs=10,
                                                filename=f'{today}/{params.model}-{params.exp_name}-bc{config.batch_size}-lr{config.lr}-dropout{config.dropout}-fc_layer{150}'+'-best_model-{epoch:02d}-{train_acc:.2f}-{val_acc:.2f}')
        checkpoint_callback = ModelCheckpoint(mode='max',
                                              save_top_k=1,
                                              monitor='val_acc',
                                              dirpath=f'checkpoint',
                                              filename=f'{today}/{params.model}-{params.exp_name}-bc{config.batch_size}-lr{config.lr}-dropout{config.dropout}-fc_layer{150}'+'-best_model-{epoch:02d}-{train_acc:.2f}-{val_acc:.2f}')
        trainer=pl.Trainer(num_nodes=1,
                           max_epochs=params.epoches,
                           accelerator="gpu",
                           log_every_n_steps=1,
                           logger=wandb_logger,
                           default_root_dir='checkpoint/',
                           callbacks=[checkpoint_callback,checkpoint_epoch_callback])
        
        trainer.fit(model=model, train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)

if __name__ == '__main__':
    params = load_config(filepath='configs/config.yaml')
    params = munchify(params)
    today = date.today()
    today = str(today.strftime("%m/%d/%y")).replace('/','',2)
    if params.init_dataset == True:
        data_preparation(params).get_even_dataset()

    main()

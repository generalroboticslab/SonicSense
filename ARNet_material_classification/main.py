import sys
import yaml
from torch import utils
from datetime import date
from munch import munchify
import pytorch_lightning as pl
from models.ARNet_material import ARNet_material
from dataset.data_preparation import data_preparation
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset.AR_material_dataset import AR_material_dataset

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def main():
    config = params
    pl.seed_everything(1,workers=True)
    model = ARNet_material(params,config)
    train_dataset = AR_material_dataset( 'train',params)
    val_dataset = AR_material_dataset('valid',params)
    train_dataloader = utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=params.num_workers)
    val_dataloader = utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=params.num_workers)

    checkpoint_callback = ModelCheckpoint(mode='max',
                                            save_top_k=1,
                                            monitor='val_acc',
                                            dirpath=f'checkpoint',
                                            filename=f'{today}/{params.model}-{params.exp_name}-bc{config.batch_size}-lr{config.lr}-dropout{config.dropout}-fc_layer{150}'+'-best_model-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}')
    trainer=pl.Trainer(num_nodes=1,
                        max_epochs=300,
                        accelerator="gpu",
                        logger=True, 
                        log_every_n_steps=1,
                        default_root_dir='checkpoint/',
                        deterministic=True,
                        callbacks=[checkpoint_callback])
    
    trainer.fit(model=model, train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)

if __name__ == '__main__':
    choice_of_config = sys.argv[1]
    params = load_config(filepath=f'configs/config_{choice_of_config}.yaml')
    params = munchify(params)
    today = date.today()
    today = str(today.strftime("%m/%d/%y")).replace('/','',2)
    if params.init_dataset == True:
        data_preparation(params).get_even_dataset()
    main()

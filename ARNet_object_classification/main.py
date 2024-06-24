import sys
import yaml
import wandb
from torch import utils
from datetime import date
from munch import munchify
import pytorch_lightning as pl
from models.ARNet_object import ARNet_object
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset.AR_object_dataset import AR_object_dataset


def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def main():    
    pl.seed_everything(1)
    model = ARNet_object(params)

    train_dataset = AR_object_dataset( 'train',params)
    val_dataset = AR_object_dataset('valid',params)

    train_dataloader = utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
    val_dataloader = utils.data.DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)
    
    logger = pl_loggers.TensorBoardLogger(save_dir=params.log_dir,name=f'{today}/{params.model}-{params.exp_name}-{params.model_inputs}-bc{params.batch_size}-lr{params.lr}-dropout{params.dropout}')

    checkpoint_callback = ModelCheckpoint(mode='max',
                                            save_top_k=1,
                                            monitor='val_acc',
                                            dirpath=f'checkpoint',
                                            filename=f'{today}/{params.model}-{params.exp_name}-{params.model_inputs}bc{params.batch_size}-lr{params.lr}-dropout{params.dropout}-fc_layer{150}'+'-best_model-{epoch:02d}-{train_acc:.2f}-{val_acc:.2f}')
    trainer=pl.Trainer(num_nodes=1,
                        max_epochs=params.epoches,
                        accelerator="gpu",
                        log_every_n_steps=1,
                        logger=logger,
                        deterministic=True,
                        default_root_dir='checkpoint/',
                        callbacks=[checkpoint_callback])
    
    trainer.fit(model=model, train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)

if __name__ == '__main__':
    choice_of_config = sys.argv[1]
    choice_of_input = sys.argv[2]
    config_path = f'configs/config_{choice_of_config}_{choice_of_input}.yaml'
    params = load_config(filepath=config_path)
    params = munchify(params)
    today = date.today()
    today = str(today.strftime("%m/%d/%y")).replace('/','',2)
    main()

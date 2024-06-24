import sys
import yaml
import torch
from torch import utils
from munch import munchify
import pytorch_lightning as pl
from models.ARnet_recon import ARnet_recon
from pytorch_lightning import loggers as pl_loggers
from dataset.AR_recon_dataset import AR_recon_dataset_s2r

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

    pl.seed_everything(1)
    for choice_of_dataset in ['valid', 'test']:
        dataset = AR_recon_dataset_s2r(choice_of_dataset, 0, params)
        dataloader = utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=params.num_workers)

        logger = pl_loggers.TensorBoardLogger(save_dir=params.log_dir)
        model = ARnet_recon(params=params)
        ckpt=torch.load(params.test_ckpt_path)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        model.freeze()

        trainer=pl.Trainer(max_epochs=params.epochs,accelerator="gpu",log_every_n_steps=1,logger=logger,default_root_dir='checkpoint/',benchmark=None)
        trainer.test(model=model, dataloaders=dataloader,verbose=True)

if __name__ == '__main__':
    main()

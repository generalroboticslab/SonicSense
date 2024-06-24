import sys
import yaml
import torch
from torch import utils
from munch import munchify
import pytorch_lightning as pl
from models.ARNet_object import ARNet_object
from dataset.AR_object_dataset import AR_object_dataset


def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def main():
    choice_of_config = sys.argv[1]
    choice_of_input = sys.argv[2]
    config_path = f'configs/config_{choice_of_config}_{choice_of_input}.yaml'
    params = load_config(filepath=config_path)
    params = munchify(params)
    pl.seed_everything(1)
    for choice_of_dataset in ['valid','test']:
        params.testing_split = choice_of_dataset.replace('original_','')
        dataset = AR_object_dataset(choice_of_dataset, params)
        dataloader = utils.data.DataLoader(dataset, batch_size=10000, shuffle=False, num_workers=params.num_workers)

        model = ARNet_object(params)
        ckpt=torch.load(params.ckpt_path)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        model.freeze()
        trainer=pl.Trainer(accelerator="gpu", devices=1,max_epochs=1,num_nodes=1)
        trainer.test(model=model,dataloaders=dataloader,verbose=True)


if __name__ == '__main__':
    main()

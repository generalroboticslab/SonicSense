import sys
import yaml
import torch
from torch import utils
from munch import munchify
import pytorch_lightning as pl
from models.ARNet_material import ARNet_material
from refine_material_predication import get_overall_accuracy
from dataset.AR_material_dataset import AR_material_dataset, AR_material_objectfolder_dataset

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

    for choice_of_dataset in ['original_valid', 'original_test']:
        params.testing_split = choice_of_dataset.replace('original_','')
        dataset = AR_material_dataset(choice_of_dataset, params)
        dataloader = utils.data.DataLoader(dataset, batch_size=10000, shuffle=False, num_workers=params.num_workers)

        model = ARNet_material(params)
        ckpt=torch.load(params.ckpt_path)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        model.freeze()

        trainer=pl.Trainer(accelerator="gpu", devices=1,max_epochs=1,num_nodes=1)
        trainer.test(model=model,dataloaders=dataloader,verbose=True)

    get_refined_results(params,config_path)

def get_refined_results(params,config_path):
    get_overall_accuracy(params,config_path).get_best_test_acc()

if __name__ == '__main__':
    main()

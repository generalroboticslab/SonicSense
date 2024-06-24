import os
import re
import random
import torch
import numpy as np
import torch.optim as Optim
from torch import optim, nn
import pytorch_lightning as pl
from tensorboardX import SummaryWriter
from chamferdist import ChamferDistance
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from visualization.visualization import plot_pcd_one_view_new
from dataset.AR_recon_dataset import AR_recon_dataset_s2r

def conv2d_bn_relu(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer
    
class ARnet_recon(pl.LightningModule):
    def __init__(self,params):
        super(ARnet_recon, self).__init__()
        self.params = params
        self.writer = SummaryWriter()
        self.latent_dim = self.params.latent_dimension
        self.num_coarse = self.params.num_of_coarse

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )

    def forward(self, points):
        B = np.shape(points)[0]
        N =  np.shape(points)[1]
        # encoder
        points_feature = self.first_conv(points.transpose(2, 1))                                       # (B,  256, N)
        points_feature_global = torch.max(points_feature, dim=2, keepdim=True)[0]                      # (B,  256, 1)
        points_feature = torch.cat([points_feature_global.expand(-1, -1, N), points_feature], dim=1)   # (B,  512, N)
        points_feature = self.second_conv(points_feature)                                              # (B, 1024, N)
        points_feature_global = torch.max(points_feature,dim=2,keepdim=False)[0]
        # decoder
        coarse = self.mlp(points_feature_global).reshape(-1, self.num_coarse, 3)                       # (B, num_coarse, 3), coarse point cloud
        return coarse.contiguous()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.params.lr,betas=(0.9, 0.999))
        scheduler = Optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)
        return [optimizer], [scheduler]

    def training_step(self,batch, batch_idx):
        p, c = batch
        coarse_pred = self.forward(p)
        coarse_pred=coarse_pred.cuda()
        c=c.cuda()
        chamferDist = ChamferDistance()
        train_loss=chamferDist(coarse_pred, c, bidirectional=True)
        index = random.randint(0, coarse_pred.shape[0] - 1)
        plot_pcd_one_view_new(os.path.join(f'image/', f'{self.params.exp_name}/'+'train/train_epoch_{:03d}.png'.format(batch_idx)),
                          [p[index].detach().cpu().numpy(),coarse_pred[index].detach().cpu().numpy(),c[index].detach().cpu().numpy()],
                          ['Input', 'Output',  'Ground Truth'])
        # Logging to TensorBoard (if installed) by default
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss
    
    def validation_step(self,batch, batch_idx):
        p, c = batch
        coarse_pred = self.forward(p)
        coarse_pred=coarse_pred.cuda()
        c=c.cuda()
        chamferDist = ChamferDistance()
        val_loss=chamferDist(coarse_pred, c, bidirectional=True)
        index = random.randint(0, coarse_pred.shape[0] - 1)
        plot_pcd_one_view_new(os.path.join(f'image/', f'{self.params.exp_name}/'+'valid/valid_epoch_{:03d}.png'.format(batch_idx)),
                          [p[index].detach().cpu().numpy(),coarse_pred[index].detach().cpu().numpy(),c[index].detach().cpu().numpy()],
                          ['Input', 'Output',  'Ground Truth'])
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch,batch_idx):
        try:
            os.makedirs(f'reconstruction_results/{self.params.exp_name}/')
        except:
            pass
        try:
            os.makedirs(f'image/{self.params.exp_name}/test/')
        except:
            pass
        
        p, c = batch
        coarse_pred = self.forward(p)
        npy = np.array(coarse_pred.cpu(),dtype=object)
        objectlist=np.load(f'dataset/{self.params.exp_name}/real_test_tapping.npy',allow_pickle=True)
        object_name=objectlist[batch_idx]
        prefix=re.findall(f"data/ARDataset/{self.params.exp_name}/tapping_points_test/\S+/", object_name)
        object_name=object_name.replace(prefix[0],'')
        print(f'reconstruction_results/{self.params.exp_name}/{object_name}')
        np.save(f'reconstruction_results/{self.params.exp_name}/{object_name}',npy)
        coarse_pred=coarse_pred.cuda()
        c=c.cuda()
        chamferDist = ChamferDistance()
        test_loss=chamferDist(coarse_pred, c, bidirectional=True)
        index = random.randint(0, coarse_pred.shape[0] - 1)
        plot_pcd_one_view_new(os.path.join(f'image/', f'{self.params.exp_name}/'+'test/test_epoch_{:03d}.png'.format(batch_idx)),
                          [p[index].detach().cpu().numpy(),coarse_pred[index].detach().cpu().numpy(),c[index].detach().cpu().numpy()],
                          ['Input', 'Output',  'Ground Truth'])
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return test_loss
        
    def setup(self, stage: None):
        self.train_dataset = AR_recon_dataset_s2r('train',self.current_epoch,self.params)
        self.val_dataset = AR_recon_dataset_s2r('valid',self.current_epoch,self.params)

    def train_dataloader(self):
        self.train_dataset = AR_recon_dataset_s2r('train',self.current_epoch,self.params)
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers,drop_last=True)
        return train_dataloader
        
    def val_dataloader(self):
        val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.params.num_workers,drop_last=True)
        return val_dataloader

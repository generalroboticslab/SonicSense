import torch
import statistics
import numpy as np
import torchmetrics
import pandas as pd
import torch.nn as nn
import seaborn as sns
import torch.optim as Optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch import optim, nn, Tensor
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import MulticlassConfusionMatrix

class ARNet_object(pl.LightningModule):
    def __init__(self,params, config=None):
        super(ARNet_object, self).__init__()
        self.params =  params
        img_channels = params.number_of_tapping_points

        # self.label_name=params.label_name
        self.label_name = [i for i in range(params.num_label)]
        self.num_classes=params.num_label
        self.points_latent_dimension = params.points_latent_dimension
        if config == None:
            self.dropout=params.dropout
            self.learning_rate=params.lr
        else:
            self.dropout=config.dropout
            self.learning_rate=config.lr

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_step_y_hats = []
        self.test_step_ys = []
        self.val_step_y_hats = []
        self.val_step_ys = []
        #tapping audio encoder
        self.dropout= nn.Dropout(p=self.dropout)

        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=16,
            kernel_size=6,
            stride=2,
            padding=0,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=150,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(150)
        self.relu3 = nn.ReLU(inplace=True)

        #tapping points encoder
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, self.points_latent_dimension, 1)
        )
        #mlp
        if self.params.model_inputs == 'audio+points':
            feature_dimension=300
        else:
            feature_dimension=150

        self.fc1 = nn.Linear(feature_dimension, 170)
        self.fc2 = nn.Linear(170, 170)
        self.fc3 = nn.Linear(170, self.num_classes)

    def forward(self, points ,audio):
        #tapping audio encoder
        audio = self.conv1(audio)
        audio = self.bn1(audio)
        audio = self.relu(audio)
        audio = self.maxpool(audio)
        audio = self.conv2(audio)
        audio = self.bn2(audio)
        audio = self.dropout(audio)
        audio = self.relu2(audio)
        audio = self.maxpool2(audio)
        audio = self.conv3(audio)
        audio = self.bn3(audio)
        audio = self.relu3(audio)
        audio_feature = torch.flatten(audio, 1)
        #tapping points encoder
        B = np.shape(points)[0]
        N =  np.shape(points)[1]
        points_feature = self.first_conv(points.transpose(2, 1))                                       # (B,  256, N)
        points_feature_global = torch.max(points_feature, dim=2, keepdim=True)[0]                      # (B,  256, 1)
        points_feature = torch.cat([points_feature_global.expand(-1, -1, N), points_feature], dim=1)   # (B,  512, N)
        points_feature = self.second_conv(points_feature)                                              # (B, 1024, N)
        points_feature_global = torch.max(points_feature,dim=2,keepdim=False)[0]
        if self.params.model_inputs == 'audio+points':
            feature = torch.cat((audio_feature, points_feature_global), 1)
        elif self.params.model_inputs == 'audio':
            feature = audio_feature
        elif self.params.model_inputs == 'points':
            feature = points_feature_global
        # mlp
        x = self.dropout(feature)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate,betas=(0.9, 0.999))
        return [optimizer]

    def training_step(self,batch,batch_idx):
        contact_points, audio, labels = batch
        contact_points = contact_points.cuda()
        audio = audio.cuda()
        labels = labels.cuda()
        outputs = self.forward(contact_points, audio)
        criterion = nn.CrossEntropyLoss()
        train_loss = criterion(outputs, labels)
        self.train_acc(outputs, labels)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.train_acc, on_epoch=True,prog_bar=True)
        return train_loss
    
    def validation_step(self,batch,batch_idx):
        contact_points, audio, labels = batch
        contact_points = contact_points.cuda()
        audio = audio.cuda()
        labels = labels.cuda()
        outputs = self.forward(contact_points, audio)
        criterion = nn.CrossEntropyLoss()
        val_loss = criterion(outputs, labels)
        self.valid_acc(outputs, labels)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.valid_acc, on_epoch=True,prog_bar=True)
        return val_loss

    def test_step(self, batch,batch_idx):
        contact_points, audio, labels = batch
        contact_points = contact_points.cuda()
        audio = audio.cuda()
        labels = labels.cuda()
        outputs = self.forward(contact_points, audio)
        self.test_acc(outputs, labels)
        self.log('test_acc', self.test_acc,on_step=False, on_epoch=True,prog_bar=True)
        #confusion matrix
        self.test_step_y_hats.append(outputs)
        self.test_step_ys.append(labels)
        y_hat = torch.cat(self.test_step_y_hats)
        y = torch.cat(self.test_step_ys)
        metric = MulticlassConfusionMatrix(num_classes=self.num_classes).cuda()
        cm=metric(y_hat,y)
        confusion_matrix_computed = cm.detach().cpu().numpy().astype(int)
        uniformed_confusion_matrix=[]
        for idx,i in enumerate(confusion_matrix_computed):
            uniformed_confusion_matrix.append([val/sum(i) for val in i])
        final_acc_list=[]
        for idx in range(len(uniformed_confusion_matrix)):
            final_acc_list.append(uniformed_confusion_matrix[idx][idx])
        final_acc=sum(final_acc_list)/len(final_acc_list)
        print('final acc = ',final_acc)
        df_cm = pd.DataFrame(uniformed_confusion_matrix,index=self.label_name,columns=self.label_name)
        plt.figure(figsize = (10,8))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Reds').get_figure()
        plt.xlabel('Predicted labels')
        plt.ylabel('True lables')
        plt.savefig(f'images/{self.params.exp_name}_{self.params.testing_split}_confusion_matrix', dpi=300)
        plt.close(fig_)
        self.loggers[0].experiment.add_figure("Test confusion matrix", fig_, self.current_epoch)
        df = pd.DataFrame(uniformed_confusion_matrix)
        df.to_excel(excel_writer = f"images/{self.params.exp_name}_{self.params.testing_split}_confusion_matrix.xlsx")
        return {'preds' : outputs, 'targets' : labels}




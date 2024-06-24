import torch
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
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import MulticlassRecall

class ARNet_material(pl.LightningModule):
    def __init__(self,params, config=None):
        super(ARNet_material, self).__init__()
        img_channels=1
        self.params =  params
        self.label_name=params.label_name
        self.num_classes=params.num_label
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
        self.dropout1= nn.Dropout(p=self.dropout)
        self.dropout2= nn.Dropout(p=self.dropout)
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
        self.fc1 = nn.Linear(150, 70)
        self.fc2 = nn.Linear(70, self.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout1(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = torch.flatten(x, 1)
        x = self.dropout2(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        # optimizer = optim.Adam(self.parameters(), lr=self.params.lr,betas=(0.9, 0.999))
        scheduler = Optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
        return [optimizer],[scheduler]

    def training_step(self,batch,batch_idx):
        image, labels = batch
        image = image.cuda()
        labels = labels.cuda()
        outputs = self.forward(image)
        criterion = nn.CrossEntropyLoss()
        train_loss = criterion(outputs, labels)
        self.train_acc(outputs, labels)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.train_acc, on_epoch=True,prog_bar=True)
        return train_loss
    
    def validation_step(self,batch,batch_idx):
        image, labels = batch
        image = image.cuda()
        labels = labels.cuda()
        outputs = self.forward(image)
        criterion = nn.CrossEntropyLoss()
        val_loss = criterion(outputs, labels)
        self.valid_acc(outputs, labels)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.valid_acc, on_epoch=True,prog_bar=True)
        return val_loss

    def test_step(self, batch,batch_idx):
        image, labels = batch
        image = image.cuda()
        labels = labels.cuda()
        outputs = self.forward(image)
        self.test_acc(outputs, labels)
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
        print('final acc among class = ',final_acc)
        df_cm = pd.DataFrame(uniformed_confusion_matrix,index=self.label_name,columns=self.label_name)
        plt.figure(figsize = (10,8))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Reds').get_figure()
        plt.xlabel('Predicted labels')
        plt.ylabel('True lables')
        plt.savefig(f'images/{self.params.exp_name}_{self.params.testing_split}_confusion_matrix', dpi=300)
        plt.close(fig_)
        self.loggers[0].experiment.add_figure("Test confusion matrix", fig_, self.current_epoch)
        #save the evaluation results
        label_prediction=[i.detach().cpu().numpy() for i in self.test_step_y_hats]
        label_gt=[i.detach().cpu().numpy() for i in self.test_step_ys]
        np.save(f'results/{self.params.exp_name}/{self.params.testing_split}/label_prediction_{self.params.testing_split}.npy',label_prediction[0])
        np.save(f'results/{self.params.exp_name}/{self.params.testing_split}/label_gt_{self.params.testing_split}.npy',label_gt[0])
        #compute metric
        recall_metric = MulticlassRecall(num_classes=self.num_classes, average='none').cuda()
        precision_metric = MulticlassPrecision(num_classes=self.num_classes, average='none').cuda()
        F1_score_metric = MulticlassF1Score(num_classes=self.num_classes,average='none').cuda()
        F1_score=F1_score_metric(y_hat,y)
        F1_score_average_metric = MulticlassF1Score(num_classes=self.num_classes,average='macro').cuda()
        F1_score_average=F1_score_average_metric(y_hat,y)
        recall = recall_metric(y_hat,y)
        precision = precision_metric(y_hat,y)
        print('torch recall', recall)
        print('torch precision ',precision)
        print('torch F1',F1_score)
        print('torch F1 average',F1_score_average)
        self.log('F1_score_average', F1_score_average,on_step=False, on_epoch=True,prog_bar=True)

        return {'preds' : outputs, 'targets' : labels}




# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # <font style="color:blue">Transfer Learning With Lightning</font>
#
# In this unit, you will learn to use PyTorch Lightning for transfer learning training.
#
#
# Once the pre-trained model is loaded, training is not different from normal model training. 
#
# So, your biggest task here is to know how to load a pretrained model with PyTorch Lightning.
#
# The `LightningModule` being a `torch.nn.Module`, with added features, just load the models from `torchvision.models`, as PyTorch Lightning models
#
# **Once the model is loaded, you  know how to use `Lightning` to train it.**

# %matplotlib inline

import matplotlib.pyplot as plt  

# +
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


# -

# ## <font style="color:green">Lightning Module</font>
#
# **We have chosen the ResNet family for transfer learning/fine-tuning.**
#
# - The ResNet family model has five  layers: `layer1`, `layer2`, `layer3`, `layer4`, and `fc`.
#
#
# - It is mandatory to replace (and re-train) the last fully connected layer (`fc`) for fine-tuning. 
#
#
# - How many more layers should be fine-tuned to get the best result is something you will know only by practically working on it each time. So, we have written a `LightningModule` class that takes `fine_tune_start` as an argument and updates the `requires_grad` parameters  of the ResNet model accordingly.

class TransferLearningWithResNet(pl.LightningModule):
    
    def __init__(self, resnet_model_name='resnet18', pretrained=True, fine_tune_start=1, num_class=3, 
                 learning_rate=0.01):
        super().__init__()
        
        self.save_hyperparameters()

        resnet = getattr(models, resnet_model_name)(pretrained=pretrained)
    
        if pretrained:
            for param in resnet.parameters():
                param.requires_grad = False
            
        if pretrained and fine_tune_start <= 1:
            for param in resnet.layer1.parameters():
                param.requires_grad = True
            
        if pretrained and fine_tune_start <= 2:
            for param in resnet.layer2.parameters():
                param.requires_grad = True
            
        if pretrained and fine_tune_start <= 3:
            for param in resnet.layer3.parameters():
                param.requires_grad = True
    
        if pretrained and fine_tune_start <= 4:
            for param in resnet.layer4.parameters():
                param.requires_grad = True    
        
            
        last_layer_in = resnet.fc.in_features
        resnet.fc = nn.Linear(last_layer_in, num_class)
        
        self.resnet = resnet

    def forward(self, x):
       
        return self.resnet(x)
    
    def training_step(self, batch, batch_idx):
        
        # get data and labels from batch
        data, target = batch

        # get prediction
        output = self(data)
        
        # calculate loss
        loss = F.cross_entropy(output, target)
        
        # get probability score using softmax
        prob = F.softmax(output, dim=1)
        
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]
        
        acc = accuracy(pred=pred, target=target, num_classes=self.hparams.num_class)
        
        
        dic = {
            'train_loss': loss,
            'train_acc': acc
        }
        

        return {'loss': loss, 'log': dic, 'progress_bar': dic}

    def training_epoch_end(self, training_step_outputs):
        # training_step_outputs = [{'loss': loss, 'log': dic, 'progress_bar': dic}, ..., 
        #{'loss': loss, 'log': dic, 'progress_bar': dic}]
        avg_train_loss = torch.tensor([x['progress_bar']['train_loss'] for x in training_step_outputs]).mean()
        avg_train_acc = torch.tensor([x['progress_bar']['train_acc'] for x in training_step_outputs]).mean()
        
        
        dic = {
            'epoch_train_loss': avg_train_loss,
            'epoch_train_acc': avg_train_acc
        }
        return {'log': dic, 'progress_bar': dic}
        
    
    def validation_step(self, batch, batch_idx):
        
        # get data and labels from batch
        data, target = batch
        
        # get prediction
        output = self(data)
        
        # calculate loss
        loss = F.cross_entropy(output, target)
        
        # get probability score using softmax
        prob = F.softmax(output, dim=1)
        
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]
        
        acc = accuracy(pred=pred, target=target, num_classes=self.hparams.num_class)
        
        
        dic = {
            'v_loss': loss,
            'v_acc': acc
        }
        
        
        return dic
    
    
    def validation_epoch_end(self, validation_step_outputs):
        # validation_step_outputs = [dic, ..., dic]
        avg_val_loss = torch.tensor([x['v_loss'] for x in validation_step_outputs]).mean()
        avg_val_acc = torch.tensor([x['v_acc'] for x in validation_step_outputs]).mean()
        
        
        dic = {
            'avg_val_loss': avg_val_loss,
            'avg_val_acc': avg_val_acc
        }
        return {'val_loss': avg_val_loss, 'log': dic, 'progress_bar': dic}
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)

# ## <font style="color:green">Lightning Data Module</font>
#
# We will be using a dataset from kaggle. [**Check it out here**](https://www.kaggle.com/ashishsaxena2209/animal-image-datasetdog-cat-and-panda).
# It consists of `1000` images each of multiple animals, with all images of a particular animal in a separate folder. We have split it into `80:20` ratio for the `train: validation`, which you can download from [**here**](https://www.dropbox.com/sh/n5nya3g3airlub6/AACi7vaUjdTA0t2j_iKWgp4Ra?dl=1).
#
# **Let's create the Lightning data module.**

# +
import os

class CatDogPandaDataModule(pl.LightningDataModule):

    def __init__(self, data_root, batch_size, num_workers):
        
        super().__init__()
        
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        mean = [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]
        
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        
        self.common_transforms = transforms.Compose([
            preprocess, 
            transforms.Normalize(mean, std)
        ])
        
        self.aug_transforms = transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.RandomGrayscale(p=0.1),
            self.common_transforms,
            transforms.RandomErasing(),
            ])
        
    def prepare_data(self):

        pass
        
    def setup(self, stage=None):
        
        train_data_path = os.path.join(self.data_root, 'training')
        val_data_path = os.path.join(self.data_root, 'validation')
        
        self.train_dataset = datasets.ImageFolder(root=train_data_path, transform=self.aug_transforms)
        
        self.val_dataset = datasets.ImageFolder(root=val_data_path, transform=self.common_transforms)


    def train_dataloader(self):
        # train loader
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers
        )
        return train_loader

    def val_dataloader(self):
        # validation loader
        test_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers
        )
        return test_loader
# -


# ## <font style="color:green">Hyperparameters</font>

# +
from argparse import ArgumentParser


def configuration_parser(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs_count', type=int, default=20)
    parser.add_argument('--data_root', type=str, default='../resource/lib/publicdata/images/cat-dog-panda')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--resnet_model_name', type=str, default='resnet18')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--fine_tune_start', type=int, default=4)
    parser.add_argument('--num_class', type=int, default=3)
    return parser



# -

# ## <font style="color:green">Training</font>

def training_validation():
    pl.seed_everything(21)    
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser = configuration_parser(parser)

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    # print(args)

    # init model
    
    model = TransferLearningWithResNet(resnet_model_name=args.resnet_model_name, 
                                       pretrained=args.pretrained, 
                                       fine_tune_start=args.fine_tune_start, 
                                       num_class=args.num_class, 
                                       learning_rate=args.learning_rate)

    data_module = CatDogPandaDataModule(data_root=args.data_root,
                                        batch_size=args.batch_size, 
                                        num_workers=args.num_workers)
    

    # most basic trainer, uses good defaults
    trainer = pl.Trainer.from_argparse_args(args,
    # fast_dev_run=True,
    max_epochs=10, 
    deterministic=True, 
    gpus=1, 
    progress_bar_refresh_rate=1, 
    early_stop_callback=True)
    
    trainer.fit(model, data_module)
    
    return model, data_module


# ### <font style="color:green">Tensorboard Dashboard</font>

# +
# To reload tensorBoard
# %load_ext tensorboard

# logs folder path
# %tensorboard --logdir=lightning_logs
# -

model, data_module = training_validation()

# # <font style="color:blue">References</font>
#
# 1. https://pytorch-lightning.readthedocs.io/en/latest/transfer_learning.html
# 1. https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# 1. https://pytorch.org/tutorials/beginner/saving_loading_models.html
# 1. https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=gEulmrbxwaYL
# 1. https://pytorch-lightning.readthedocs.io
# 1. https://github.com/PyTorchLightning/pytorch-lightning
# 1. https://www.youtube.com/watch?v=QHww1JH7IDU
# 1. https://pytorch-lightning.readthedocs.io/en/latest/
# 1. https://www.youtube.com/channel/UC8m-y0yAFJpX0hRvxH8wJVw/featured

#!/usr/bin/env python
# coding: utf-8

'''
mkdir logs pair_samples pretrained_weights
sbatch --error=logs/genesis_lung.out --output=logs/genesis_lung.out run.sh /data/jliang12/zzhou82/holy_grail None
'''

import warnings
warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import torch
from torchsummary import summary
import sys
import argparse
from utils import *
import unet3d
from config import models_genesis_config
from tqdm import tqdm

print("torch = {}".format(torch.__version__))

parser = argparse.ArgumentParser(description='main')
parser.add_argument('--gpu', dest='gpu', default=None, type=str, help="gpu index")
parser.add_argument('--data', dest='data', default=None, type=str, help="the direction of dataset")
parser.add_argument('--weights', dest='weights', default=None, type=str, help="load the pre-trained models")
args = parser.parse_args()

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

conf = models_genesis_config(args=args)
conf.display()

training_generator = generate_pair_custom_loss(conf, status='train')
validation_generator = generate_pair_custom_loss(conf, status='val')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = unet3d.UNet3D()
model.to(device)

print("Total CUDA devices: ", torch.cuda.device_count())

summary(model, (1,conf.input_rows,conf.input_cols,conf.input_deps), batch_size=-1)

if conf.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), conf.lr, momentum=0.9, weight_decay=0.0, nesterov=False)
elif conf.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), conf.lr)
else:
    raise

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(conf.patience * 0.6), gamma=0.5)

mse_loss = nn.MSELoss()                                                         
mae_loss = nn.L1Loss()
# to track the training loss as the model trains
train_losses = []
mse_metric = []                                                                 
mae_metric = []                                                                 
# to track the validation loss as the model trains                              
valid_losses = []                                                               
mseval_losses = []                                                              
maeval_losses = []                                                              
# to track the average training loss per epoch as the model trains              
avg_train_losses = []                                                           
# to track the average validation loss per epoch as the model trains            
avg_valid_losses = []                                                           
avg_l2train_losses = []                                                         
avg_l1train_losses = []                                                         
avg_l2val_losses = []                                                           
avg_l1val_losses = []
best_loss = 100000
intial_epoch = 0
num_epoch_no_improvement = 0
sys.stdout.flush()

if conf.weights != None:
    checkpoint=torch.load(conf.weights)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    intial_epoch=checkpoint['epoch']
    print("Loading weights from ",conf.weights)
sys.stdout.flush()


for epoch in range(intial_epoch, conf.nb_epoch):
    scheduler.step(epoch)
    model.train()
    for iteration in range(conf.steps_per_epoch):
        image, gt = next(training_generator)
        gt = np.repeat(gt, conf.nb_class, axis=1)
        image, gt = torch.from_numpy(image).float().to(device), torch.from_numpy(gt).float().to(device)
        pred = model(image)
        loss = custom_loss(gt, pred)                                             
        loss2 = mse_loss(pred,gt[:int((gt.size()[0])/2)])                       
        loss3 = mae_loss(pred,gt[:int((gt.size()[0])/2)])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(round(loss.item(), 2))                              
        mse_metric.append(round(loss2.item(), 2))                               
        mae_metric.append(round(loss3.item(), 2))
        if (iteration + 1) % 5 ==0:
            print('Epoch [{}/{}], iteration {}, Loss: {:.6f}, MSELoss: {:.6f}, MAELoss: {:.6f}'
                .format(epoch + 1, conf.nb_epoch, iteration + 1, np.average(train_losses), np.average(mse_metric), np.average(mae_metric)))
            sys.stdout.flush()

    with torch.no_grad():
        model.eval()
        print("validating....")
        for i in range(conf.validation_steps):
            x, y = next(validation_generator)
            y = np.repeat(y,conf.nb_class,axis=1)
            image,gt = torch.from_numpy(x).float(), torch.from_numpy(y).float()
            image = image.to(device)
            gt = gt.to(device)
            pred = model(image)
            loss = custom_loss(gt,pred)                                             
            loss2 = mse_loss(pred,gt[:int((gt.size()[0])/2)])                       
            loss3 = mae_loss(pred,gt[:int((gt.size()[0])/2)])
            valid_losses.append(loss.item())                                        
            mseval_losses.append(round(loss2.item(), 2))                            
            maeval_losses.append(round(loss3.item(), 2))
    
    #logging
    train_loss = np.average(train_losses)                                         
    valid_loss = np.average(valid_losses)                                         
    l2train_loss = np.average(mse_metric)                                       
    l1train_loss = np.average(mae_metric)                                       
    l2val_loss = np.average(mseval_losses)                                      
    l1val_loss = np.average(maeval_losses)                                      
    avg_train_losses.append(train_loss)                                         
    avg_valid_losses.append(valid_loss)                                         
    avg_l2train_losses.append(l2train_loss)                                     
    avg_l1train_losses.append(l1train_loss)                                     
    avg_l2val_losses.append(l2val_loss)                                         
    avg_l1val_losses.append(l1val_loss)
    print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}, MSE val is {:.4f}, MAE val is {:.4f}, MSE train is {:.4f}, MAE train is {:.4f}".format(epoch+1,valid_loss,train_loss,l2val_loss,l1val_loss,l2train_loss,l1train_loss))
    train_losses = []
    valid_losses = []
    mse_metric = []                                                               
    mae_metric = []                                                               
    mseval_losses = []                                                            
    maeval_losses = []
    if valid_loss < best_loss:
        print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
        best_loss = valid_loss
        num_epoch_no_improvement = 0
        #save model
        torch.save({
            'epoch': epoch+1,
            'state_dict' : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        },os.path.join(conf.model_path, "Genesis_Chest_CT"+str(epoch+1)+".pt"))
        print("Saving model ",os.path.join(conf.model_path,"Genesis_Chest_CT"+str(epoch+1)+".pt"))
    else:
        print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,num_epoch_no_improvement))
        num_epoch_no_improvement += 1
    if num_epoch_no_improvement == conf.patience:
        print("Early Stopping")
        break
    sys.stdout.flush()

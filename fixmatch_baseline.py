from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, models, transforms
from collections import OrderedDict
import matplotlib.pyplot as plt
import time
import os
import copy

from dataset.oxfordiiitpet import DATASET_GETTERS



data_dir = "./data/multiclass"
input_size = 224
num_classes = 37
batch_size = 32
num_epochs = 100 # A maximum number, not intended to be reached
fc_type = "double" # Supports "single" and "double", number of fully connected layers, connected by ReLU - default = 'single'
optimizer_type = "adam" # Supports "adam" and "sgd" - default = 'adam'
momentum = 0.9 # Set if using sgd
initial_lr_fc = 1e-3
initial_lr_ft = 1e-5
patience_scheduler = 3
patience_stopping = 8

#Number of labeled samples:
num_labeled = 1850  #has to be a multiple of 37
num_unlabeled = 3680  #used for # of batches
#Paper, section 4:
mu = 7
threshold = 0.95
lambda_u = 1

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_parameter_requires_grad(model, n_layers, reversed_params=True):
    count = 0
    if reversed_params:
        param_list = reversed(list(model.named_parameters()))
    else:
        param_list = list(model.named_parameters())
    for name, param in param_list:
        if count >= n_layers:
            param.requires_grad = False
        elif "conv" in name or "bn" in name:
            count += 1
            param.requires_grad = True

def print_params(model):
    for name, param in model.named_parameters():
        print(name, param.size(), param.requires_grad)
        
def test_model(model, testloader, criterion):
    model.eval()
    with torch.no_grad(): #not training
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(testloader.dataset)
        epoch_acc = running_corrects.double() / len(testloader.dataset)
    return epoch_loss, epoch_acc

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, patience=999, scheduler=None):
    since = time.time()
    train_loss_history = []
    train_loss_x_history = []
    train_loss_u_history = []
    test_acc_history = []
    test_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = np.inf
    patience_count = 0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        # Each epoch trains and then tests
        ### not sure about the testing

        model.train()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter() #a bit easier to use, equal to running losses
        running_corrects = 0
        # Iterate over batches of labeled and unlabeled data.
        labeled_trainloader, unlabeled_trainloader = dataloaders['train']
        for lab_batch, unlab_batch in zip(labeled_trainloader, unlabeled_trainloader):
            inputs_x, targets_x = lab_batch 
            (inputs_u_w, inputs_u_s), _ = unlab_batch
            inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).to(device)
            targets_x = targets_x.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward, track history
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                ###not sure if should 'interleave', or what it is for
                logits = model(inputs)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)

                #labeled loss, assuming criterion is cross entropy
                targets_x = targets_x.type(torch.LongTensor).to(device)
                loss_x = criterion(logits_x, targets_x)

                #pseudo label with weak augmentation, no temperature used
                max_probs, targets_u = torch.max(logits_u_w, dim=-1) ### not sure if only max(logits) is needed
                mask = max_probs.ge(threshold).float() #greater or equal than threshold

                #unlabeled loss, pseudolabel and mask
                targets_u = targets_u.type(torch.LongTensor).to(device)
                loss_u = (criterion(logits_u_s, targets_u) * mask).mean()
                #_, preds = torch.max(outputs, 1) ###how to measure acc?

                loss = loss_x + lambda_u * loss_u

                #if phase == 'train':
                loss.backward()
                optimizer.step()
            # statistics, average is correct for equally sized batches
            losses.update(loss.item())
            losses_x.update(loss_x.item())
            losses_u.update(loss_u.item())
            #running_corrects += torch.sum(preds == labels.data)

        epoch_loss = losses.avg
        train_loss_history.append(epoch_loss)
        #epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
        #train_acc_history.append(epoch_acc)
        epoch_loss_x = losses_x.avg
        train_loss_x_history.append(epoch_loss_x)
        epoch_loss_u = losses_u.avg
        train_loss_u_history.append(epoch_loss_u)

        #if epoch+1 == num_epochs: #testing ### not needed?
            # load best model weights to return and use for testing
            #model.load_state_dict(best_model_wts)
 
        test_loss, test_acc = test_model(model, dataloaders['test'], criterion)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)
        print('Loss: {:.4f} Test Acc: {:.4f}'.format(epoch_loss, test_acc))
        if scheduler:                  
            scheduler.step(test_loss)
            
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        if test_loss <= best_loss:
            patience_count = 0
            best_loss = test_loss
        else:
            patience_count +=1
        
        if patience_count >= patience:
            print('Training stopped, Early stopping triggered')
            model.load_state_dict(best_model_wts)
            model.eval()
            test_loss, test_acc = test_model(model, dataloaders['test'], criterion)
            print('test Loss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            
            history = {'test_acc' : test_acc_history, 'test_loss' : test_loss_history, 
                       'train_loss' : train_loss_history, 'train_loss_x' : train_loss_x_history,
                       'train_loss_u' : train_loss_u_history}
            return model, history

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    history = {'test_acc' : test_acc_history, 'test_loss' : test_loss_history, 
            'train_loss' : train_loss_history, 'train_loss_x' : train_loss_x_history,
            'train_loss_u' : train_loss_u_history}
    return model, history

def initialize_model(num_classes, n_layers, fc_type='single', use_pretrained=True, verbose=False):
    # Use Resnet34 as model
    model_ft = models.resnet34(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, n_layers)
    num_ftrs = model_ft.fc.in_features
    
    if fc_type == 'double':
        model_ft.fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_ftrs,num_ftrs)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(num_ftrs,num_classes)),
        ('output', nn.LogSoftmax(dim=1))
        ]))
    else:
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    if verbose:
        print_params(model_ft)

    return model_ft


if __name__ == '__main__':
    # Initialize the model for this run
    model_ft = initialize_model(num_classes, 0, fc_type=fc_type, use_pretrained=True, verbose=True)
    print("Model initialized")
    
    # Loading datasets through oxfordiiitpet.py
    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS["oxfordiiitpet"](num_labeled, "datasets")

    ### deleted sampler from dataloader
    labeled_trainloader = torch.utils.data.DataLoader(
        labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True)

    unlabeled_trainloader = torch.utils.data.DataLoader(
        unlabeled_dataset,
        batch_size=batch_size*mu,
        shuffle=True,
        num_workers=4,
        drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True)

    # Create training and validation dataloaders
    dataloaders_dict = {'train': [labeled_trainloader, unlabeled_trainloader],
                         'test': test_loader}
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_ft = model_ft.to(device)
    
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            
    if optimizer_type == 'sgd':
        optimizer_fc = optim.SGD(params_to_update, lr=initial_lr_fc, momentum=momentum)
    else:
        optimizer_fc = optim.Adam(params_to_update, lr=initial_lr_fc)
        
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer_fc, 'min', patience = patience_scheduler)
    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_fc, num_epochs=num_epochs, patience=patience_stopping, scheduler=scheduler)
    
    # Next phase of training : second half of all layers + fc
    set_parameter_requires_grad(model_ft, 48)
    
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            
    if optimizer_type == 'sgd':
        optimizer_ft = optim.SGD(params_to_update, lr=initial_lr_ft, momentum=momentum)
    else:
        optimizer_ft = optim.Adam(params_to_update, lr=initial_lr_ft)
        
    scheduler = ReduceLROnPlateau(optimizer_ft, 'min', patience = patience_scheduler)
        
    model_ft2, hist2 = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, patience=patience_stopping, scheduler=scheduler)
    
    # Next phase of training : first half of all layers - fc
    
    set_parameter_requires_grad(model_ft, 48, reversed_params=False)
    
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            
    if optimizer_type == 'sgd':
        optimizer_ft = optim.SGD(params_to_update, lr=initial_lr_ft, momentum=momentum)
    else:
        optimizer_ft = optim.Adam(params_to_update, lr=initial_lr_ft)
        
    scheduler = ReduceLROnPlateau(optimizer_ft, 'min', patience = patience_scheduler)
        
    model_ft3, hist3 = train_model(model_ft2, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, patience=patience_stopping, scheduler=scheduler)
    
    # Final phase of training : fc again
    params_to_update = []
    for name, param in list(model_ft3.named_parameters()):
        if "fc" in name:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False
            
    if optimizer_type == 'sgd':
        optimizer_ft = optim.SGD(params_to_update, lr=initial_lr_ft, momentum=momentum)
    else:
        optimizer_ft = optim.Adam(params_to_update, lr=initial_lr_ft)
        
    scheduler = ReduceLROnPlateau(optimizer_ft, 'min', patience = patience_scheduler)
        
    model_ft4, hist4 = train_model(model_ft3, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, patience=patience_stopping, scheduler=scheduler)

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


### Parameters
data_dir = "./data/multiclass"
# data_dir = "../input/oxford-pets-classified-image-dataset/data/multiclass"
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

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, patience=999, scheduler=None):
    since = time.time()
    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = np.inf
    patience_count = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        phases = ['train', 'val']
        if (epoch+1) == num_epochs:
            phases.append('test')
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                if phase == 'test':
                    # load best model weights to return and use for testing
                    model.load_state_dict(best_model_wts)
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                print('LR : {} '.format(optimizer.param_groups[0]['lr']))
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
                                         
                if scheduler:                  
                    scheduler.step(epoch_loss)
                                         
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch_acc = epoch+1
                    best_model_wts = copy.deepcopy(model.state_dict())
                if epoch_loss <= best_loss:
                    patience_count = 0
                    best_model_loss = copy.deepcopy(model.state_dict())
                    best_epoch_loss = epoch+1
                    best_loss = epoch_loss
                else:
                    patience_count +=1
                if patience_count >= patience:
                    print('Training stopped, Early stopping triggered')
                    model.load_state_dict(best_model_loss)
                    model.eval()
                    # Iterate over data.
                    running_loss = 0.0
                    running_corrects = 0
                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        with torch.set_grad_enabled(False):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            _, preds = torch.max(outputs, 1)
                            running_loss += loss.item() * inputs.size(0)
                            running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                    test_acc_loss = epoch_acc
                    print('test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
                    print('Best model wrt loss, epoch = {}'.format(best_epoch_loss))
                    model.load_state_dict(best_model_wts)
                    model.eval()
                    # Iterate over data.
                    running_loss = 0.0
                    running_corrects = 0
                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        with torch.set_grad_enabled(False):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            _, preds = torch.max(outputs, 1)
                            running_loss += loss.item() * inputs.size(0)
                            running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                    test_acc_acc = epoch_acc
                    print('test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
                    print('Best model wrt acc, epoch = {}'.format(best_epoch_acc))
                    
                    if test_acc_acc < test_acc_loss:
                        test_acc = test_acc_loss
                        model.load_state_dict(best_model_loss)
                    else:
                        test_acc = test_acc_acc

                    time_elapsed = time.time() - since
                    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                    history = {'val_acc' : val_acc_history, 'val_loss' : val_loss_history, 'train_acc' : train_acc_history, 'train_loss' : train_loss_history, 'test_acc' : test_acc}
                    return model, history
                
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    history = {'val_acc' : val_acc_history, 'val_loss' : val_loss_history, 'train_acc' : train_acc_history, 'train_loss' : train_loss_history, 'test_acc' : test_acc}
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
    print('Model initialized')
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4779, 0.4434, 0.3940], [0.2677, 0.2630, 0.2697]) # Mean and std calculated for pet dataset
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.4779, 0.4434, 0.3940], [0.2677, 0.2630, 0.2697]) # Mean and std calculated for pet dataset
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.4779, 0.4434, 0.3940], [0.2677, 0.2630, 0.2697]) # Mean and std calculated for pet dataset
        ]),
    }

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}

    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}
    
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
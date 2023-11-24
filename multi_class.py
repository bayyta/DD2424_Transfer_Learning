from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

data_dir = "./data/multiclass"
num_classes = 37
batch_size = 32
num_epochs = 15
layers_to_train = 0 # Number of layers to finetune other than the final layer. So [0 <= layers_to_train <= 17].

def set_parameter_requires_grad(model, n_layers):
    count = 0
    for name, param in reversed(list(model.named_parameters())):
        if count >= n_layers:
            param.requires_grad = False

        if "conv" in name:
            count += 1

def print_params(model):
    for name, param in model.named_parameters():
        print(name, param.size(), param.requires_grad)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    return model, val_acc_history

def initialize_model(num_classes, n_layers, use_pretrained=True):
    # Use Resnet18 as model
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, n_layers)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    #print_params(model_ft)

    return model_ft, input_size


if __name__ == '__main__':
    # Initialize the model for this run
    model_ft, input_size = initialize_model(num_classes, layers_to_train, use_pretrained=True)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4779, 0.4434, 0.3940], [0.2677, 0.2630, 0.2697]) # Mean and std calculated below for pet dataset
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.4779, 0.4434, 0.3940], [0.2677, 0.2630, 0.2697])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.4779, 0.4434, 0.3940], [0.2677, 0.2630, 0.2697])
        ]),
    }

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}

    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}


    """
        Calculate mean and std for train+val data. Should be without any transforms on the images.
    """
    # -------------------------------------------
    """
    tot_pixels = 0
    mean_channels = torch.zeros(3, dtype=float)

    # Get mean for each channel
    for s in ['train', 'val']:
        for x in image_datasets[s]:
            w, h = x[0].shape[1], x[0].shape[2]

            tot_pixels += (w * h)
            mean_channels += torch.sum(x[0].reshape((3, w * h)), dim=1)
            #for i in range(3):
                #pxls_channels[i].extend(x[0][i, :, :].reshape((w * h)).tolist())

    mean_channels /= tot_pixels
    mean_channels = mean_channels.reshape((3, 1))

    std = torch.zeros(3, dtype=float)

    # Loop again to calculate std, dataset too big to load into memory at once
    for s in ['train', 'val']:
        for x in image_datasets[s]:
            w, h = x[0].shape[1], x[0].shape[2]

            std += torch.sum((x[0].reshape((3, w * h)) - mean_channels)**2, dim=1)

    std /= (tot_pixels - 1)
    std = std**0.5
    print("Mean:", mean_channels.ravel())
    print("Std:", std)
    """
    # -------------------------------------------


    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    optimizer_ft = optim.Adam(params_to_update, lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)
    

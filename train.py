import numpy as np
import pandas as pd
import torch
import workspace_utils
import argparse
import json
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict
from PIL import Image
import torch.nn.functional as F

def get_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('dir', type = str, default = 'flowers/', help="The folder with the pet images.")
    parser.add_argument('--arch', type = str, default = 'vgg13', help="The Model Architecture to use (vgg13, vgg16 or vgg19).")
    parser.add_argument('--learning_rate', type = float, default = 0.01, help="Learning rate for the model's Adam optimizer.")
    parser.add_argument('--hidden_units', type = int, default = 512, help="Number of hidden units in the hidden layer for the neural network.")
    parser.add_argument('--epochs', type = int, default = 5, help="Number of epochs to go through when training..")
    parser.add_argument('--gpu', action='store_true', help="Number of epochs to go through when training..")
    
    return parser.parse_args()

def load_data(data_dir):
    train_dir = data_dir + 'train'
    valid_dir = data_dir + 'valid'
    test_dir = data_dir + 'test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    validate_transforms = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validate_data = datasets.ImageFolder(valid_dir, transform=validate_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloaders = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validateloaders = torch.utils.data.DataLoader(validate_data, batch_size=64)
    testloaders = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return trainloaders, validateloaders, testloaders, train_data

def create_model(model_name, hidden_units, learning_rate, device):
    if model_name == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        model = models.vgg19(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.25),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)
    return model, criterion, optimizer

def get_device(should_use_gpu):
    device = 'cpu'
    if should_use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return device

def train(model, criterion, optimizer, epochs, trainloaders, validateloaders, device, model_name, train_data):
    steps = 0
    running_loss = 0
    print_every = 5

    with workspace_utils.active_session():
        for epoch in range(epochs):
            for inputs, labels in trainloaders:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logprobs = model.forward(inputs)
                loss = criterion(logprobs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validateloaders:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logprobs = model.forward(inputs)
                            batch_loss = criterion(logprobs, labels)

                            valid_loss += batch_loss.item()

                            probs = torch.exp(logprobs)
                            top_p, top_class = probs.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss/len(validateloaders):.3f}.. "
                          f"Validation accuracy: {accuracy/len(validateloaders):.3f}")
                    running_loss = 0
                    model.train()
                    
    save_model(model, optimizer, epochs, model_name, train_data)
                    
def save_model(model, optimizer, epochs, model_name, train_data):
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'state_optim': optimizer.state_dict(),
                  'epochs': epochs,
                  'model_name': model_name,
                  'class_to_idx': model.class_to_idx,
                  'classiffier': model.classifier,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')

args = get_input_args()
trainloader, validateloader, testloader, train_data = load_data(args.dir)
device = get_device(args.gpu)
model, criterion, optimizer = create_model(args.arch, args.hidden_units, args.learning_rate, device)
train(model, criterion, optimizer, args.epochs, trainloader, validateloader, device, args.arch, train_data)

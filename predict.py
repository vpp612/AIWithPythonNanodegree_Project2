import argparse
import numpy as np
import pandas as pd
import torch
import argparse
import json
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict
from PIL import Image
import torch.nn.functional as F

def get_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image_path', type=str, default='flowers/test/74/image_01151.jpg',
                        help="Image file path to use for prediction.")
    parser.add_argument('checkpoint_path', type=str, default='checkpoint.pth',
                        help="Training checkpoint file path to load.")
    parser.add_argument('--top_k', type=int, default=5, help="Number of predicted classes to return.")
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help="File that contains the category names.")
    parser.add_argument('--gpu', action='store_true', help="Number of epochs to go through when training..")
    
    return parser.parse_args()

def map_labels(label_mapping_file_path):
    with open(label_mapping_file_path, 'r') as f:
        categories_to_name = json.load(f)
        
    return categories_to_name

def get_device(should_use_gpu):
    device = 'cpu'
    if should_use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return device

def load_checkpoint(checkpoint_file_path, device):
    checkpoint = torch.load(checkpoint_file_path)
    if checkpoint['model_name'] == 'vgg13':
        model = models.vgg13()
    elif checkpoint['model_name'] == 'vgg16':
        model = models.vgg16()
    else:
        model = models.vgg19()
        
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classiffier']
    model.load_state_dict(checkpoint['state_dict'])
    
    model.to(device)
    return model

def process_image(image):
    img = Image.open(image)
    img.thumbnail((256, 256))
    img = img.crop((16, 16, 240, 240)) #crop a 224x224 region
    
    np_image = np.array(img)
    np_image = np.true_divide(np_image, 255)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image - mean) / std
    
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def predict(image_path, model, topk=5):
    processed_image = process_image(image_path)
    processed_image = torch.from_numpy(processed_image).float().to(device)
    processed_image.unsqueeze_(0)
    
    log_probs = model(processed_image)
    probs = torch.exp(log_probs)
    top_p, top_class = probs.topk(topk, dim=1)
    
    return top_p, top_class

def print_probabilities(top_p, top_class, model):
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_class = [idx_to_class[c.item()] for c in top_class[0]]
    top_class = [categories_to_name[c] for c in top_class]
    top_p = [p.item() for p in top_p[0]]
    
    for c in range(len(top_class)):
        print('{}- {}%'.format(top_class[c], top_p[c]*100))

# main execution
args = get_input_args()
categories_to_name = map_labels(args.category_names)
device = get_device(args.gpu)
model = load_checkpoint(args.checkpoint_path, device)
top_p, top_class = predict(args.image_path, model, args.top_k)
print_probabilities(top_p, top_class, model)

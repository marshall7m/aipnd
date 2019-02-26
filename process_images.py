from torchvision import datasets, transforms, models 
import torch
from PIL import Image
import numpy as np

def process_train_test(train_dir, valid_dir, test_dir):
    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomRotation(15),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                                      ])
    test_valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                                      ])
    
    train_data = datasets.ImageFolder(train_dir, transform= train_transforms) 
    valid_data = datasets.ImageFolder(valid_dir, transform= test_valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform= test_valid_transforms) 
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size= 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size= 32, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size= 32, shuffle = True)
    return trainloader, validloader, testloader, train_data

def process_image(image):
    im = Image.open(image)
    
    width, height = im.size
    if width > height:
        im.thumbnail((500, 256))
    else:
        im.thumbnail((256, 500))
        
    width, height = im.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = left + 224
    bottom = top + 224
    
    im = im.crop((left, top, right, bottom))
    im.thumbnail((224,224))
    np_im = np.array(im)/255 
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_im = (np_im - mean)/std
    np_im.transpose((2, 0, 1))
    
    return np_im


import torch
from torchvision import models
from torch import nn
from torch import optim
from collections import OrderedDict
import re
def load_pretrained_model(model):
    if model == 'vgg16':
        model = models.vgg16(pretrained = True)
        pretrained_output = model.classifier[0].in_features
        return model, pretrained_output
    elif model == 'resnet18':
        model = models.resnet18(pretrained = True)
        pretrained_output = model.fc.in_features
        return model, pretrained_output
    elif model == 'alexnet':
        model = models.alexnet(pretrained = True)
        pretrained_output = model.classifier[1].in_features
        return model, pretrained_output
    else:
        print('Please choose one of the three models: vgg16, resnet18, alexnet')
        
def classifier_param(model, arch, pretrained_output, hidden_units, output_units, drop_p):
   for param in model.parameters():
        param.requires_grad = False
        
   input_size = pretrained_output
   hidden_size = hidden_units
   output_size = output_units
        
   classifier = nn.Sequential(OrderedDict([
              ('fc1', nn.Linear(input_size, hidden_size[0])),
              ('relu1', nn.ReLU()),
              ('drop1', nn.Dropout(p = drop_p)),
              ('fc2', nn.Linear(hidden_size[0], hidden_size[1])),
              ('relu2', nn.ReLU()),
              ('drop2', nn.Dropout(p = drop_p)),
              ('fc3', nn.Linear(hidden_size[1], output_size)),
              ('output', nn.LogSoftmax(dim=1))]))
   if arch == 'resnet18':
       model.fc = classifier
       return model
   else:
       model.classifier = classifier
       return model


def deep_train(model, arch, trainloader, validloader, learning_rate, epochs, print_every, device):
    device = torch.device("cuda:0" if torch.cuda.is_available() and device == 'gpu' else 'cpu')
        
    epochs = epochs
    print_every = print_every
    steps = 0
    criterion = nn.NLLLoss()
    if arch == 'resnet18':
        optimizer = optim.Adam(model.fc.parameters(), lr= learning_rate)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr= learning_rate)
    
    model.to(device)
    
    for e in range(epochs):
        running_loss = 0
        for ii, (images, labels) in enumerate(trainloader):
            steps += 1
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                
                model.eval()
                
                with torch.no_grad():
                    test_loss, accuracy = test_model(model, validloader, criterion, 'gpu')
                    
                print('Epoch: {}/{} ... '.format(e+1, epochs),
                      'Running Loss: {:.3f} ... '.format(running_loss/print_every),
                      'Test Loss: {:.3f} ... '.format(test_loss/len(validloader)),
                      'Accuracy: {:.3f}'.format(accuracy/len(validloader)))
                
                model.train()
                running_loss = 0
    return model
    
def test_model(model, validloader, criterion, device):
    device = torch.device("cuda:0" if torch.cuda.is_available() and device == 'gpu' else 'cpu')
    accuracy = 0
    test_loss = 0
    
    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels == ps.max(dim= 1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy


def model_accuracy_test(model, testloader, device):   
    device = torch.device("cuda:0" if torch.cuda.is_available() and device == 'gpu' else 'cpu')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} test images: {:.1f}'.format(total, (100 * correct / total)))
    

def model_checkpoint(checkpoint_label, model, arch, train_data, epochs, learning_rate, drop_prob):  
    if arch == 'resnet18':
        optimizer = optim.Adam(model.fc.parameters(), lr= learning_rate)
        classifier = getattr(model, 'fc')
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr= learning_rate)
        classifier = getattr(model, 'classifier')
    torch.save({
            'state_dict': model.state_dict(),
            'arch': arch,
            'classifier': classifier,
            'optimizer': optimizer.state_dict(),
            'epochs': epochs,
            'class_to_idx': train_data.class_to_idx,
            }, 'checkpoint.tar')
        




                
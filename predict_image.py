import torch
from torchvision import models
from train import classifier_param
from train import load_pretrained_model
from torch import optim
from process_images import process_image
import json
import re
import numpy as np

def model_load(path):
    checkpoint = torch.load(path)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_to_idx']
    model.eval()
    
    return model

def predict(image_path, category_names, model, top_k, device):
    device = torch.device("cuda:0" if torch.cuda.is_available() and device == 'gpu' else 'cpu')
    
    image = process_image(image_path)
    tensor_im = torch.from_numpy(image)
    tensor_im = tensor_im.permute(2, 0, 1).type(torch.cuda.FloatTensor)
    tensor_im = tensor_im.unsqueeze(0)
    
    with torch.no_grad():
        model.to(device)
        tensor_im.to(device)
        output = model.forward(tensor_im)
        probs = torch.exp(output)
    
    probabilities, idx = torch.topk(probs, top_k)
    idx = idx.cpu().numpy()[0]
    probabilities = probabilities.cpu().numpy()[0]
    classes = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [classes[each] for each in idx]
    
    with open(category_names, 'r') as f:
        category_names = json.load(f)
    label_idx = re.split('/', image_path)
    label_idx = label_idx[2]
    label = category_names[label_idx]   
    
    predicted_labels = [category_names[i] for i in top_classes]
    round_prob = list(np.around(np.array(probabilities),2))
    predict_dict = dict(zip(predicted_labels, round_prob))
    
    print('Predicted Image Label: ', label)
    print('Top ' + str(top_k) + ' Probabilities')
    for k,v in predict_dict.items():
        print(str(k) + ': '+ str(v))
    
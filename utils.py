# Imports here
import numpy as np
import torch
from torchvision import datasets,transforms
import torchvision.models as models
from collections import OrderedDict
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import json

arch = {'densenet121':1024,'vgg16':25088}
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, dropout=0.5):
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])       
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])        
        self.output = nn.Linear(hidden_layers[-1], output_size)       
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)        
        x = self.output(x)  
        x = F.log_softmax(x, dim=1)
        return x
def load_data(data_path):
    batch_size = 64
    data_dir = data_path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    train_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                                 ])
    data_transforms = transforms.Compose([transforms.Resize(256),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                              ])
    train_data = datasets.ImageFolder(train_dir ,transform = train_data_transforms)
    test_data = datasets.ImageFolder(test_dir,transform = data_transforms)
    valid_data = datasets.ImageFolder(valid_dir,transform = data_transforms)

    train_dataloader = torch.utils.data.DataLoader(train_data,batch_size = batch_size,shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data,batch_size=32)
    valid_dataloader = torch.utils.data.DataLoader(valid_data,batch_size=32)
    return train_data,train_dataloader,test_dataloader,valid_dataloader
def build_model(output_size, hidden_layers, learning_rate=0.001,architecture="vgg16",gpu="gpu",dropout=0.5):
    #pre-trained model VGG16
    if architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    
    #froze parameters
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = Network(arch[architecture], output_size, hidden_layers, dropout=0.5)   
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)   
    if gpu == "gpu":
        model.to('cuda')
    return model,criterion,optimizer
# calculate validation loss and accuracy
def do_accuracy(model, dataloader, criterion,gpu):
    valid_loss = 0    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader: 
            if gpu == "gpu":
                images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model.forward(images)
            valid_loss += criterion(outputs, labels).item()           
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return valid_loss/total,correct / total
def train_model(model,train,validate,criterion,optimizer,epochs,gpu):
    for e in range(epochs):
        for ii, (images, labels) in enumerate(train): 
            if gpu == "gpu":
                images, labels = images.to('cuda'), labels.to('cuda')        
            optimizer.zero_grad()        
            # Forward and backward
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()                
        #check validation loss and accuracy for each epoch        
        with torch.no_grad():
            model.eval()
            validate_loss, accuracy = do_accuracy(model, validate, criterion,gpu)
            print("Epoch:{}".format(e+1),"Validation Loss: {:.3f}.. ".format(validate_loss),                
                    "Validation Accuracy: {:.3f}".format(accuracy)) 
def save_checkpoint(model,architecture,save_dir,train_data,learning_rate,core):            
    # TODO: Save the checkpoint 
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'gpu':core,
              'arch': architecture,
              'lr':learning_rate,
              'output_size': 102,
              'class_to_idx': model.class_to_idx,
              'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
              'state_dict': model.state_dict()}
    torch.save(checkpoint, save_dir)
def load_model(path):
    checkpoint = torch.load(path)
    model,_,_ = build_model(checkpoint['output_size'],
                    checkpoint['hidden_layers'],
                    checkpoint['lr'],
                    checkpoint['arch'],
                    checkpoint['gpu'])
    model.load_state_dict(checkpoint['state_dict'])    
    return model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    width, height = img.size
    re_size =256
    shortest_side = min(width, height)
    new_width = int((width/shortest_side)*re_size)
    new_height = int((height/shortest_side)*re_size)
    img = img.resize((new_width,new_height))
    
  
    #center crop
    crop_size = 224
    w0 = (new_width - crop_size) / 2
    h0 = (new_height - crop_size) / 2
    cropped_image = img.crop((w0, h0, w0 + crop_size, h0 + crop_size))
    np_image = np.array(cropped_image) / 255.0
    
    #normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225]) 
    image_norm = (np_image - mean) / std
    trp_img = image_norm.transpose((2, 0, 1))
    
    return trp_img
def predict(image_path, model, core,category_file,topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''  
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    image = torch.from_numpy(image).float()
    image.unsqueeze_(0)    
    with torch.no_grad():
        output = model.forward(image.to('cuda'))
    with open(category_file, 'r') as f:
       cat_to_name = json.load(f)   
    probabilities = F.softmax(output.data,dim=1)
    prob,classes = probabilities.topk(topk)
    classes_5 = [cat_to_name[str(ID)] for ID in np.array(classes[0])]
    return prob,classes_5




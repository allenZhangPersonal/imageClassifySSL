import torch
from torchvision import models, transforms
from torch import nn
import torch.nn.functional as F

class pretrain_model(torch.nn.Module): # Used for rotnet
  def __init__(self):
    super().__init__()
    ### You can modify your model here
    ### Please visit https://pytorch.org/vision/stable/models.html for different models
    ### Resnet18 is just a simple network acting as a placeholder
    self.resnet = models.resnet18(num_classes=2000)
    self.fc1 = nn.Linear(2000,512)
    self.fc2 = nn.Linear(512,128)
    self.fc3 = nn.Linear(128,4)
  def forward(self, x):
    output = self.resnet(x)
    output = F.relu(output)
    output = self.fc1(output)
    output = F.relu(output)
    output = self.fc2(output)
    output = F.relu(output)
    output = self.fc3(output)
    sigmoid = nn.Sigmoid()
    output = sigmoid(output)
    return output

# Downstream model on image classification 
class downstream_model(torch.nn.Module):
  def __init__(self, output_layer = None): # For resnet18, output layer will manipulate which layer for transfer task
    super().__init__()
    ### Add architecture here
    self.resnet = models.resnet18(num_classes=2000)
    self.output_layer = output_layer
    self.layers = list(self.resnet._modules.keys()) # Converts layer names to a list
    self.layer_count = 0
    for l in self.layers:
        if l != self.output_layer:
            self.layer_count += 1 # Counts number of layers until we reach the same layer name
        else:
            break
    for i in range(1,len(self.layers)-self.layer_count):
        self.resnet._modules.pop(self.layers[-i]) # Pop layers from resnet depending on layer count
        
    self.resnet = nn.Sequential(self.resnet._modules) # Create sequential neural network
    self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # Averages pools the output to 256x1x1
    self.fc4 = nn.Linear(256,800) # Utilizes fully connected layer for 800 class image classifcation
    ### Add architecture here
  def forward(self, x):
    output = self.resnet(x)
    output = self.avgpool(output).squeeze(2).squeeze(2) # Reduces dimension
    output = self.fc4(output)
    return output
	
def get_pretrain_model():
	return pretrain_model()

def get_downstream_model():
	return downstream_model('layer3'): # We use the 3rd layer for resnet18
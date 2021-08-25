# Generic python libraries
import os
from datetime import date
# Pytorch specific libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from PIL import Image
# Custom .py files
from dataloader import CustomDataset
import globals
from models import get_pretrain_model, get_downstream_model
from time import time

def loadData(rootDir, task):

	pre_labels = True if task == "pretext" else False
	# Normalize RGB images
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                                 std=[0.229, 0.224, 0.225]) # For RGB images

	transformations = [] # Transformations for 0, 90, 180, and 270 degrees of rotation plus other augmentations
	for i in range(4): # 4 rotations
		transformations.append(transforms.Compose([
			transforms.Resize((256, 256)),
			transforms.RandomAffine((i*90,i*90)), # handles rotation
			transforms.ColorJitter(brightness=(0.5,1.5), contrast=(0.5,1.5), saturation=(0.5,1.5), hue=(-0.1,0.1)),
			transforms.ToTensor(),  # convert PIL to Pytorch Tensor
			normalize,
		]))

	if not pre_labels:
		# More augmentations for downstream task
		transformations.append(transforms.Compose([
			transforms.Resize((256, 256)),
		    transforms.RandomAffine((0,360)),
		    transforms.RandomHorizontalFlip(),
		    transforms.GaussianBlur(kernel_size = (5,5), sigma=(0.1, 2.0)),
		    transforms.ToTensor(),  # convert PIL to Pytorch Tensor
			normalize,
		]))
		transformations.append(transforms.Compose([
			transforms.RandomCrop(72),
		    transforms.Resize((256, 256)),
		    transforms.RandomAffine((0,360)),
		    transforms.RandomHorizontalFlip(),
		    transforms.ColorJitter(brightness=(0.5,1.5), contrast=(0.5,1.5), saturation=(0.5,1.5), hue=(-0.1,0.1)),
		    transforms.ToTensor(),  # convert PIL to Pytorch Tensor
		]))

	# Custom datasets
	labeledset = []
	unlabeledset = []
	validationset = []

	# Populate list with each dataset
	for i in range(len(transformations)):
		# From labeled training set
		labeledset.append(CustomDataset(root=rootDir, split="labeled", transform=transformations[i], rotNet=pre_labels, rotDegree=i))
		if pre_labels:
			# From unlabeled set
			unlabeledset.append(CustomDataset(root=rootDir, split="unlabeled", transform=transformations[i], rotNet=pre_labels, rotDegree=i))
			# From validation set
			validationset.append(CustomDataset(root=rootDir, split="validation", transform=transformations[i], rotNet=pre_labels, rotDegree=i))

	if not pre_labels:
		val_transform = transforms.Compose([
		    transforms.Resize((256, 256)),
		    transforms.ToTensor(),  # convert PIL to Pytorch Tensornormalize,
		    normalize,
		])
		validationset.append(CustomDataset(root=rootDir, split="validation", transform=val_transform, rotNet=pre_labels, rotDegree=0))

	# Validation loader
	valset = torch.utils.data.ConcatDataset(validationset) # Concatenates the list into a single dataset
	valloader = torch.utils.data.DataLoader(valset, batch_size=256, shuffle=False, num_workers=4)

	# Trainset loader
	trainset = torch.utils.data.ConcatDataset(labeledset.extend(unlabeledset)) # Concatenates the list into a single dataset
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)

	return trainloader, valloader

def initiate_train(trainloader, importFile, pipeLine):
	logFile = open(globals.log_file_path, 'a')
	model = get_pretrain_model().to(globals.device)
	accuracy = 0 # initial accuracy
	# Get folder path
	folderPath = os.path.join(globals.weights, globals.folder_name)
	if importFile:
		directorylist = next(os.walk(globals.weights))[1] # Retrieves only folder names
		if globals.folder_name in directoryList:
			weightFiles = os.listdir(folderPath)
			maxWeight = 0
			for weights in weightFiles:
				if "pretext_train_weights" in weights:
					accuracy = weights.split("_")[3].split(".") # Retrieve only accuracy part
					accuracy = ".".join(accuracy[:2]) # Remove .pt extension
					flAccuracy = float(accuracy)
					if flAccuracy > maximum:
						maximum = flAccuracy
			filename = "pretext_train_weights_{:.2f}.pth".format(maximum) # file name of pretrained weight
			state_dict = torch.load(os.path.join(folderPath, filename)) # load the weight into our state_dict
			print("Continue training file: {}".format(filename), file=logFile)
			accuracy = maximum # Current maximum accuracy
		else:
			os.mkdir(folderPath) # Make directory for today's date
			state_dict = None
	if pipeLine == "predownstream_train": # rotnet accuracy is different from classification
		max_accuracy = 0
		accuracy = 0
		pipeLine = "downstream_train"
	else:
		max_accuracy = accuracy
	if pipeLine != "pretext_train":
		print("Begin downstream training model", file=logFile, flush=True)
	else:
		print("Begin pretext training model", file=logFile, flush=True)
	save_dict = None
	for i in range(len(globals.num_epochs_list)):
		accuracy, state_dict = train_model(trainloader, globals.num_epochs_list[i], globals.lr_list[i], globals.stop_list[i]
			model, accuracy, "pipeLine", folderPath, pipeLine != "pretext_train", state_dict)
		if max_accuracy < accuracy:
			max_accuracy = accuracy
			save_dict = state_dict
		print("Current maximum accuracy: {}\n".format(max_accuracy), file=logFile)
	print("Our maximum accuracy: {}".format(max_accuracy), file=logFile)
	print("Finish {} model".format("pipeLine"), file=logFile, flush=True)
	logFile.close()
	return model, save_dict, folderPath
# dataloader - dataloader for dataset
# NUM_EPOCH - number of epoch for current learning rate
# learning_rate - current learning rate
# early_stop - how many epoch is go for early stop condition 
# model - the actual model
# prev_accuracy - maximum accuracy from input weights or previous learning
# pipeLine - downstream or pretrain
# folderPath - path to save weights
# freeze - condition to freeze gradient or not
# load_dict - the dictionary from previous loading or previous learning rate
def train_model(dataloader, NUM_EPOCH, learning_rate, early_stop, model, prev_accuracy, pipeLine, folderPath, freeze=False, load_dict=None, m=0.9, decay=5*(10**-4)):
	model_dict = model.state_dict()
	logFile = open(globals.log_file_path, 'a')
	# Check if weight is imported
	if load_dict:
		import_weights = load_dict
		import_weights = {k: v for k, v in import_weights.items() if k in model_dict} # Extract only same keys in both dictionaries
		model_dict.update(import_weights) # Update the weights
		model.load_state_dict(model_dict, strict=False) # Load the weight

	if freeze: # freezes gradient
		size = 0
		for params in model.parameters():
			params.require_grad = False
			size += 1
		index = 0
		for params in model.parameters():
			if(size - index <= globals.freeze_layers): # This depends on dense layer on top of resnet
				params.require_grad = True
			index += 1
	## Use cross entropy and SGD, can change if needed
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=m, weight_decay=decay)
    print('Start {} for learning rate: {}'.format(learning_rate, pipeLine), file=logFile, flush=True)

    ## Start training
    model.train()
    max_epoch_ind = 0 # epoch of maximum accuracy
    max_accuracy = prev_accuracy # Gets from input
    model_dict = model.state_dict() # Initial weights
    for epoch in range(NUM_EPOCH):
    	current_time = time()
    	running_loss = 0.0
    	total = 0
    	correct = 0
    	total_loss = 0.0
    	for i, data in enumerate(dataloader):
    		# get the inputs; data is a list of [inputs, labels]
    		inputs, labels = data
    		inputs, labels = inputs.to(globals.device), labels.to(globals.device)
    		# Output of network
			outputs = net(inputs)
            loss = criterion(outputs,labels)
            # predictions
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            total_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches can change depending on number of samples
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10), file=logFile, flush=True)
                running_loss = 0.0

        print(f"Accuracy per epoch: {(100 * correct / total):.2f}%", file=logFile, flush=True) ### prints per epoch
    	print('Total loss per epoch: [%d, %5d] loss: %.3f' % (epoch, i + 1, total_loss), file=logFile, flush=True)
    	current_accuracy = (100 * correct / total)
    	weight = model.state_dict()
    	torch.save(weight, os.path.join(folderPath, "{}_weights_{:.2f}.pth".format(pipeLine, current_accuracy))) # Saving model

    	# Checking if current epoch accuracy is higher than our maximum accuracy
        if ((100 * correct / total) > max_accuracy):
            max_accuracy = (100 * correct / total)
            max_epoch_ind = epoch
            weights = model.state_dict() # Use these weights
            print("New maximum model saved for learning rate: {} on Accuracy: {:.2f}".format(learning_rate,max_accuracy), file=logFile)  
        else:
        	print("Model saved each epoch for learning rate: {} on Accuracy: {:.2f}".format(learning_rate, current_accuracy), file=logFile)
        ### Early stopping, continuing learning doesn't do anything for our model
        if epoch - max_epoch_ind > early_stop:
            print("Early stop triggered check log for information.", file=logFile) # Early stop triggered
            break
        print("This epoch took: {:.2f} seconds".format(time() - current_time), file=logFile)
    print('Finished {} learning rate: {}'.format(pipeLine,learning_rate), file=logFile, flush=True)
    logFile.close()
    return max_accuracy, weights

# Validate model on validation set for pretraining every epoch
# model - actual model
# valloader - validation dataset
def validateModel(valloader, model, weights, folderPath):
	logFile = open(globals.log_file_path, 'a')
    print("Start validating on validation set", file=logFile)
    model.load_state_dict(weights, strict=False) # Load the weight
    with torch.no_grad(): # No training only evaluating
        sumOfLabelsVal = 0
        total = 0
        correct = 0
        i = 0
        for data in valloader:
            images, labels = data
            images, labels = images.to(globals.device), labels.to(globals.device)
            sumOfLabelsVal += labels.sum().item()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        current_val_accuracy = (100 * correct / total)
        print("Correct: {}, Total: {}, Current evaluation accuracy: {:.2f}".format(correct, total, current_val_accuracy), file=logFile, flush=True)
        print("Sanity check, sum of all labels for validation set: {}".format(sumOfLabelsVal), file=logFile)
        weight = model.state_dict()
        pipeLine = "validation"
        torch.save(weight, os.path.join(folderPath, "{}_weights_{:.2f}.pth".format(pipeLine, current_val_accuracy))) # saving model
        print("Model saved for validation. Accuracy: {:.2f}".format(current_val_accuracy), file=logFile)
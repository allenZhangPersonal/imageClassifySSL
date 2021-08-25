import os
import torch
from pathlib import Path
from datetime import date

def initialize():
	global root
	global weights
	global split
	global label_filename
	global num_epochs_list
	global lr_list
	global stop_list
	global freeze_layers
	global device
	global folder_name
	global log_file_path
	root = os.path.join(Path(os.getcwd()).parent.absolute(),'dataset') # Directory with labeled, unlabeled and validation sets
	weights = os.path.join(Path(os.getcwd()).parent.absolute(),'weights') # Directory with weights
	# Placeholder for split and filename of labels for that split
	split = "labeled"
	label_filename = "labeled_label_tensor.pt"
	# Decaying learning rate
	num_epochs_list = [20,20,15,10] # List of epoch for different learning rate
	lr_list = [0.1,0.01,0.001,0.0001] # Different learning rates (decaying method)
	stop_list = [7,7,5,3] # change this for different stops (If the maximum accuracy for validation hasn't changed for this amount of epoch, we continue with decay learning rate)
	# For transfer learning
	freeze_layers = 6 # How many layers to not freeze to be precise
	# CPU or GPU
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # Check your graphics setting to find exact cuda index
	# For folder_name change this to custom folder name for weight to retrieve
	today = date.today() # Today's date
	folder_name = today.strftime("%Y_%m_%d") # YYYY_MM_DD
	log_file_path = os.path.join(Path(os.getcwd()).parent.absolute(),'logs')
	log_file_path = os.path.join(log_file_path, "{}.txt".format(folder_name))
def setFileName():
	global pretrain_weight
	pretrain_weight = "pretext_train_weights_99.99.pt"
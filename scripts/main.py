import globals 
import train_model

if __name__ == "__main__": 
	globals.initialize()
	trainloader, valloader = train_model.loadData(globals.root, "pretext_train")
	# Change to True if you have a pretrained weight file to continue training 
	train_model.initiate_train(trainloader, False, "pretext") 

	# Downstream training
	trainloader, valloader = train_model.loadData(globals.root, "downstream")
	# Should have weight since we pretrained in pretext task
	model, weights = train_model.initiate_train(trainloader, True, "predownstream_train") 

	# Validate model
	train_model.validateModel(valloader, model, weights, folderPath)

# Image classification with RotNet pretext training and Data augmentation
## Please use the code below to create your conda environment and activate your environment <br/>
conda env create -f environment.yml <br/>
conda activate myenv <br/>

## Please note that this code only acts as a starter for image classification on easy tasks, if you require a more intricate task, please utilize parts of the code for your own customization purposes <br/>

To utilize the code, run: [main.py](https://github.com/allenZhangPersonal/imageClassifySSL/blob/main/scripts/main.py). <br/>
Most customizations can be changed within [globals.py](https://github.com/allenZhangPersonal/imageClassifySSL/blob/main/scripts/globals.py) where global variables are defined. <br/>
[dataloader.py](https://github.com/allenZhangPersonal/imageClassifySSL/blob/main/scripts/dataloader.py) consists simple loader logic for images <br/>
[models.py](https://github.com/allenZhangPersonal/imageClassifySSL/blob/main/scripts/models.py) contains a simple resnet18 model for pretext and downstream, please use your own model for your purposes. Note that in my experience, wider and deeper NNs are required for greater number of classes to classify. <br/>
[train_model.py](https://github.com/allenZhangPersonal/imageClassifySSL/blob/main/scripts/train_model.py) loads the data, trains the model and validates the model. <br/>


## If there are any questions or bugs, please submit a pull request and I will take a look at it. Thank you!

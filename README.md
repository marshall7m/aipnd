# AI Programming with Python Project

## Description:
Project code for Udacity's AI Programming with Python Nanodegree program. In this project, I develop code for an image classifier built with PyTorch, then convert it into a command line application. 

## Data:
The dataset consisted of jpg files of plants and their corresponding true label of what species of plants they are. There are a total of 102 possible lables. 

## Processing:

* All pictures were resized to 224 x 224 and each of their color channels were normalized to a value between -1 to 1 centered at 0. These transforms are needed for the pretrained network to accept the input data.
* Data augmentation was used for the training data to help the model generalize better and improve overall performance. 

## Model Architecture: (VGG-16 with 3 FC Layers)

VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (fc1): Linear(in_features=25088, out_features=5000, bias=True)
    (relu1): ReLU()
    (drop1): Dropout(p=0.2)
    (fc2): Linear(in_features=5000, out_features=500, bias=True)
    (relu2): ReLU()
    (drop2): Dropout(p=0.2)
    (fc3): Linear(in_features=500, out_features=102, bias=True)
    (output): LogSoftmax()
  )
)

Model's Hyperparameters:

* Learning Rate: .001
* Epochs: 12
* Optimizer: Adam Optimizer
* Loss Fn: NLLLoss

Results:

* Accuracy: 81.4

Command Line Application:

* ``
* Using predict.py file to single prediction:
```python predict.py --input_image flowers/valid/1/image_06739.jpg --category_names cat_to_name.json --checkpoint checkpoint.tar --top_k 5 --device gpu```




# -*- coding: utf-8 -*-
"""Práctica Deep Learning - Transfer_learning_tutorial.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DrAP9iORyJrqjwnZyF9jZ6hn54LITT2P
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

"""
# Transfer Learning for Computer Vision Tutorial
**Author**: [Sasank Chilamkurthy](https://chsasank.github.io)

In this tutorial, you will learn how to train a convolutional neural network for
image classification using transfer learning. You can read more about the transfer
learning at [cs231n notes](https://cs231n.github.io/transfer-learning/)_

Quoting these notes,

    In practice, very few people train an entire Convolutional Network
    from scratch (with random initialization), because it is relatively
    rare to have a dataset of sufficient size. Instead, it is common to
    pretrain a ConvNet on a very large dataset (e.g. ImageNet, which
    contains 1.2 million images with 1000 categories), and then use the
    ConvNet either as an initialization or a fixed feature extractor for
    the task of interest.

These two major transfer learning scenarios look as follows:

-  **Finetuning the convnet**: Instead of random initialization, we
   initialize the network with a pretrained network, like the one that is
   trained on imagenet 1000 dataset. Rest of the training looks as
   usual.
-  **ConvNet as fixed feature extractor**: Here, we will freeze the weights
   for all of the network except that of the final fully connected
   layer. This last fully connected layer is replaced with a new one
   with random weights and only this layer is trained.
"""

# License: BSD
# Author: Sasank Chilamkurthy

#from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# upload external file before import
#from google.colab import files

#model_path = '../models/model_ft.pth'
model_path = '../models/model_ft-2.pth'
#model_path = '../models/model_conv.pth'

cudnn.benchmark = True
plt.ion()   # interactive mode

#from google.colab import drive
#drive.mount('/content/gdrive')

#!unzip /content/gdrive/MyDrive/INSE_SmartRecycling/data/hymenoptera_data.zip -d data/

"""## Load Data

We will use torchvision and torch.utils.data packages for loading the
data.

The problem we're going to solve today is to train a model to classify
**ants** and **bees**. We have about 120 training images each for ants and bees.
There are 75 validation images for each class. Usually, this is a very
small dataset to generalize upon, if trained from scratch. Since we
are using transfer learning, we should be able to generalize reasonably
well.

This dataset is a very small subset of imagenet.

.. Note ::
   Download the data from
   [here](https://download.pytorch.org/tutorial/hymenoptera_data.zip)
   and extract it to the current directory.


"""

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256), #FFM para test COCO
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#data_dir = '../data/EXERCISE/'
data_dir = '../data/DATASET/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
#class_names = image_datasets['train'].classes
#class_names = ['brontosaurus', 'elephant', 'rhino', 'stegosaurus']
class_names = ['bottle', 'mouse', 'pencilcase', 'raspberry']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""### Visualize a few images
Let's visualize a few training images so as to understand the data
augmentations.


"""

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])

"""## Testing the model

Now, let's write a general function to test a model. 

"""
  
def test_model(model, criterion, phase):
    since = time.time()

    test_acc = 0.0
           
    model.eval()   # Set model to evaluate mode

    print('')
    print('[Testing model]')
    
    running_loss = 0.0
    running_corrects = 0
    num_images_processed = 0
    
    # Iterate over data.
    for inputs, labels in dataloaders[phase]:
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        num_images_processed += inputs.size(0)

        print(f'Processing {num_images_processed} images of {dataset_sizes[phase]}...')
        
        # forward
        # We do not need to track history of gradients
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
            
    test_loss = running_loss / dataset_sizes[phase]
    test_acc = running_corrects.double() / dataset_sizes[phase]

    time_elapsed = time.time() - since
    print('')
    print(f'[Test results]')
    print(f'Test complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Test Acc: {test_acc:4f}')
    print(f'Test Loss: {test_loss:4f}')
    print()

    return test_acc


"""### Visualizing the model predictions

Generic function to display predictions for a few images



"""

def visualize_model(model, num_images=6):
    was_training = model.training
    print(f'was_training = {was_training}')
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])
                print(f'predicted: {class_names[preds[j]]} ({preds[j]})')

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.waitforbuttonpress(5)
                    plt.close(fig)
                    return
        model.train(mode=was_training)

"""## Finetuning the convnet

Load a pretrained model and reset final fully connected layer.



"""

#model_ft = models.resnet18(pretrained=True)
model_ft = models.mobilenet_v2(pretrained=True)

#num_ftrs = model_ft.fc.in_features #FFM para resnet

# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
#model_ft.fc = nn.Linear(num_ftrs, len(class_names)) #FFM para COCO
model_ft.classifier[1] = nn.Linear(1280, len(class_names)) #FFM para MobileNet
print(model_ft)

#model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

"""### Evaluate
It should take around 30 seconds for 40 sample images
"""

#model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

model_ft = models.mobilenet_v2(pretrained=True)
model_ft.classifier[1] = nn.Linear(1280, len(class_names)) #FFM para MobileNet_v2 fine-tuneado con Colab
model_ft.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model_ft.eval()
test_model(model_ft, criterion, 'val')
visualize_model(model_ft, num_images=4)
plt.ioff()
plt.show()


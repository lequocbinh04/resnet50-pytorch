import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import argparse
import time

def train_model(model, criterion, optimizer, num_epochs=3):
  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)

    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()
      else:
        model.eval()

      running_loss = 0.0
      running_corrects = 0

      for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

      epoch_loss = running_loss / len(image_datasets[phase])
      epoch_acc = running_corrects.double() / len(image_datasets[phase])

      print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                  epoch_loss,
                                                  epoch_acc))
  return model

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument('--train', required=True,
                  help='path to folder train')
  ap.add_argument('--valid', required=False, 
                  help='path to folder valid') 
  ap.add_argument('-o','--output', required=True,
                  help='path to save model trained')                 
  args = ap.parse_args()
  train_path = args.train
  valid_path = args.valid if args.valid else train_path

  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

  data_transforms = {
      'train':
      transforms.Compose([
          transforms.Resize((224,224)),
          # transforms.RandomGrayscale(p=0.1),
          # transforms.RandomAffine(0, shear=5, scale=(0.8,1.2)),
          # transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.8, 1.5), saturation=0),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize
      ]),
      'val':
      transforms.Compose([
          transforms.Resize((224,224)),
          transforms.ToTensor(),
          normalize
      ]),
  }

  image_datasets = {
      'train': 
      datasets.ImageFolder(train_path, data_transforms['train']),
      'val': 
      datasets.ImageFolder(valid_path, data_transforms['train'])
  }

  dataloaders = {
      'train':
      torch.utils.data.DataLoader(image_datasets['train'],
                                  batch_size=4,
                                  shuffle=True,
                                  num_workers=0),  
      'val':
      torch.utils.data.DataLoader(image_datasets['train'],
                                  batch_size=4,
                                  shuffle=False,
                                  num_workers=0)
  }
  class_names = image_datasets['train'].classes
  dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
  print(class_names)
  print(dataset_sizes)
  inputs, classes = next(iter(dataloaders['train']))
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model_ft = models.resnet50(pretrained=True)
  num_ftrs = model_ft.fc.in_features
  model_ft.fc = nn.Linear(num_ftrs, len(class_names))

  model_ft = model_ft.to(device)
  model_ft.cuda()
  model_ft.eval()
  criterion = nn.CrossEntropyLoss()
  # Observe that all parameters are being optimized
  optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

  # Decay LR by a factor of 0.1 every 7 epochsScratch
  # Final Thoughts and Where to Go Next
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
  model_trained = train_model(model_ft, criterion, optimizer_ft, num_epochs=5)
  torch.save(model_trained.state_dict(), args.output + '/model_'+str(time.time() % 1000)+'.pth')
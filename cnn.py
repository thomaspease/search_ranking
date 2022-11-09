from doctest import OutputChecker
from dataset import data_loaders, dataset_sizes
from torchvision import models
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch
import time
import copy
# from torch.utils.tensorboard import SummaryWriter


class CNN(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = torch.nn.Sequential(
      torch.nn.Conv2d(3,8,9, padding=1),
      torch.nn.ReLU(),
      torch.nn.Conv2d(8,8,9, padding =1),
      torch.nn.ReLU(),
      torch.nn.Flatten(),
      torch.nn.Linear(107648, 10),
      torch.nn.Softmax()
    )

  def forward(self, features):
    return self.layers(features)

def train(model, criterion, optimiser, scheduler, epochs=25):
  since = time.time()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    print('-' * 10)

    for phase in ['train', 'val']:
      if phase == 'train':
          model.train()  # Set model to training mode
      else:
          model.eval()   # Set model to evaluate mode

      running_loss = 0.0
      running_corrects = 0

      # Iterate over data.
      for batch in data_loaders[phase]:
        optimiser.zero_grad()
        inputs = batch['image']
        labels = batch['category']

        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          # Here the blank space is where the raw max numbers will be left, the preds will get the class/index
          _, preds = torch.max(outputs, 1) 
          loss = criterion(outputs, labels)

          if phase == 'train':
            loss.backward()
            optimiser.step()

        # Below you are multiplying by inputs.size because loss.item is the mean loss accross the number of items in the batch. As you are going to divide by the total items in the dataset you need a non meaned number.
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

      if phase == 'train':
        scheduler.step()

      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double() / dataset_sizes[phase]

      print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
  
  time_elapsed = time.time() - since
  print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
  print(f'Best val Acc: {best_acc:4f}')

  torch.save(best_model_wts, 'model.pt')

  model.load_state_dict(best_model_wts)
  return model

model = models.resnet50()
num_ftrs = model.fc.in_features

model.fc = torch.nn.Linear(num_ftrs, 10)
criterion = torch.nn.CrossEntropyLoss()

optimiser = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimiser, step_size=7, gamma=0.1)

model_conv = train(model, criterion, optimiser,
                         exp_lr_scheduler)
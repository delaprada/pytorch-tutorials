from random import shuffle
from turtle import forward
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES'] ='2'

# device config
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 784 # 28 * 28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST(手写数字集)
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# see one batch of data
examples = iter(train_loader)
samples, labels = examples.next()

print(samples.shape, labels.shape) # torch.Size([100, 1, 28, 28]) torch.Size([100])

for i in range(6):
  plt.subplot(2, 3, i + 1)
  plt.imshow(samples[i][0], cmap='gray')
# plt.show()

class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNet, self).__init__()
    self.l1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.l2 = nn.Linear(hidden_size, num_classes)
  
  def forward(self, x):
    out = self.l1(x)
    out = self.relu(out)
    out = self.l2(out)
    return out # 此处不需要做 softmax，因为后续使用 torch 的 crossEntropy 方法时会自动应用 softmax 方法

model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    # reshape image first, [100, 1, 28, 28] -> [100, 784]
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)

    model.to(device) # 需要将 model 也 to(device)，否则会报错：RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!

    # forward
    outputs = model(images)

    loss = criterion(outputs, labels)

    # backwards
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 100 == 0:
      print(f'epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}' )

# test
with torch.no_grad():
  n_correct = 0
  n_samples = 0

  for images, labels in test_loader:
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)
    outputs = model(images)

    # value, index
    _, predictions = torch.max(outputs, 1)
    n_samples += labels.shape[0]
    n_correct += (predictions == labels).sum().item()
  
  acc = 100.0 * n_correct / n_samples
  print(f'accuracy = {acc}')

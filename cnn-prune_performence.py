import os
import time
import torch
import torch.nn as nn
import torchvision
import stat
import torchvision.transforms as transforms
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from memory_profiler import memory_usage


# Setup device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
amounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Constants
EPOCHS = 1
BATCH_SIZE = 50
LEARNING_RATE = 0.001
DOWNLOAD_MNIST = not (os.path.exists('./mnist/') and os.listdir('./mnist/'))

# Data preparation
train_transform = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=train_transform, download=DOWNLOAD_MNIST)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=train_transform)
test_x = test_data.data.unsqueeze(1).type(torch.FloatTensor)[:2000] / 255.0
test_y = test_data.targets[:2000]
test_x, test_y = test_x.to(device), test_y.to(device)

# Model definition
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x

# Model initialization

def measure_performance(model, data_loader):
    model.eval()
    start_time = time.time()
    total, correct = 0, 0
    
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs, _ = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    elapsed_time = time.time() - start_time
    memory = max(memory_usage(-1, interval=1, timeout=1))
    accuracy = 100 * correct / total
    return elapsed_time, memory, accuracy

for amount in amounts:
  cnn = CNN().to(device)
  optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
  loss_func = nn.CrossEntropyLoss()
  prune.l1_unstructured(cnn.conv1[0], name='weight', amount=amount)
  # Training with performance tracking
  times, memories, accuracies = [], [], []
  #save_path='./result/performance_{amount}.jpg'
  # if not os.path.exists(save_path):#检查目录是否存在
  #               os.makedirs(save_path)#如果不存在则创建目录
  for epoch in range(EPOCHS):
      for step, (b_x, b_y) in enumerate(train_loader):
          b_x, b_y = b_x.to(device), b_y.to(device)
          outputs, _ = cnn(b_x)
          loss = loss_func(outputs, b_y)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          if step % 50 == 0:
              elapsed_time, memory, accuracy = measure_performance(cnn, train_loader)
              times.append(elapsed_time)
              memories.append(memory)
              accuracies.append(accuracy)
              print(f'Epoch: {epoch} | Step: {step} | Loss: {loss.item()} | Accuracy: {accuracy}% | Time: {elapsed_time}s | Memory: {memory}MB | Prune Amount: {amount}')
              # Visualization of performance metrics

      plt.figure(figsize=(12, 4))
      plt.subplot(1, 3, 1)
      plt.plot(times, label='Time (s)')
      plt.title('Inference Time')
      plt.xlabel('Steps')
      plt.ylabel('Time (s)')
      plt.legend()

      plt.subplot(1, 3, 2)
      plt.plot(memories, label='Memory Usage (MB)')
      plt.title('Memory Usage')
      plt.xlabel('Steps')
      plt.ylabel('Memory (MB)')
      plt.legend()

      plt.subplot(1, 3, 3)
      plt.plot(accuracies, label='Accuracy (%)')
      plt.title('Accuracy')
      plt.xlabel('Steps')
      plt.ylabel('Accuracy (%)')
      plt.legend()

      plt.tight_layout()
      plt.savefig(f'./result/performance_{amount}.jpg')
      #plt.show()
      
      




import os
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from memory_profiler import memory_usage


#选择GPU进行推理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#对比不同剪枝量 
amounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

#训练参数
EPOCHS = 1
BATCH_SIZE = 50
LEARNING_RATE = 0.005
DOWNLOAD_MNIST = not (os.path.exists('./mnist/') and os.listdir('./mnist/'))

#训练数据准备
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
#训练数据加载
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

#测试数据准备
test_data = torchvision.datasets.MNIST(
    root='./mnist/', 
    train=False, 
    transform=transforms.ToTensor(),)

#测试数据处理
test_x = test_data.data.unsqueeze(1).type(torch.FloatTensor)[:2000] / 255.0
test_y = test_data.targets[:2000]
test_x, test_y = test_x.to(device), test_y.to(device)

#模型定义
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

#性能评估函数
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

if __name__ == '__main__':
  #每训练一次更新一次剪枝量并生成性能数据进行对比
  for amount in amounts:

    cnn = CNN().to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()

    #剪枝前的权重
    torch.save(cnn.state_dict(), f'./result/cnn_l1_initial_weights_{amount}.pth')

    #l1_unstructured剪枝
    prune.l1_unstructured(cnn.conv1[0], name='weight', amount=amount)
    prune.remove(cnn.conv1[0], 'weight')
    
    #剪枝后的权重
    torch.save(cnn.state_dict(), f'./result/cnn_l1_pruned_weights_{amount}.pth')

    times, memories, accuracies = [], [], []

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
    plt.savefig(f'./result/l1_performance_{amount}.jpg')
  
      
      




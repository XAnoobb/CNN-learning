import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

EPOCH = 10
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

if not (os.path.exists('./mnist/') and os.listdir('./mnist/')):
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

#torchvision.datasets.MNIST(...): 从 torchvision 的数据集中加载MNIST数据集。train=False 表示加载的是测试集而非训练集。
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
#unsqueeze(1): 在数据张量中增加一个维度，MNIST数据默认为 [2000, 28, 28]（2000个样本，每个样本28x28像素），unsqueeze(1) 使其变为 [2000, 1, 28, 28]，增加的维度代表通道数，因为卷积层通常期望数据具有明确的通道维度。
#type(torch.FloatTensor): 将数据类型转换为浮点数，这是因为神经网络模型通常在浮点数上进行计算。
#[:2000]: 这表示从测试集中仅取前2000个样本进行测试。
#/ 255.0: 将像素值从 [0, 255] 归一化到 [0.0, 1.0] 范围，这有助于模型训练的稳定性和收敛性。
#test_y = test_data.targets[:2000]: 从测试集中获取前2000个样本的标签
test_x = test_data.data.unsqueeze(1).type(torch.FloatTensor)[:2000] / 255.0
test_y = test_data.targets[:2000]
test_x, test_y = test_x.to(device), test_y.to(device)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # 第一层卷积层
        self.conv1 = torch.nn.Sequential(
            # 卷积层，使用16个滤波器（输出通道），每个滤波器大小为5x5
            # 输入通道为1（单通道图像，如灰度图），步长为1，填充为2
            # 填充为2是为了保证输入和输出的空间尺寸相同，即输出也为28x28
            torch.nn.Conv2d(1, 16, 5, 1, 2),
            # ReLU激活函数，增加非线性，帮助网络学习复杂的模式
            torch.nn.ReLU(),
            # 最大池化层，使用2x2的窗口进行池化
            # 这会将特征图的空间维度降低，比如从28x28降低到14x14
            torch.nn.MaxPool2d(2),
        )
        
        # 第二层卷积层
        self.conv2 = torch.nn.Sequential(
            # 卷积层，使用32个滤波器，滤波器大小为5x5
            # 输入通道为16（匹配上一层的输出通道），步长为1，填充为2
            # 输出的空间尺寸仍然为14x14（因为填充为2）
            #第二个卷积层的输出通道数设为32是为了使模型能够捕获更复杂的特征，同时维持一个合理的计算效率
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            # ReLU激活函数
            torch.nn.ReLU(),
            # 最大池化层，使用2x2的窗口
            # 这一步再次降低特征图的空间维度，从14x14降低到7x7
            torch.nn.MaxPool2d(2),
        )
        
        # 输出层
        self.out = torch.nn.Linear(32 * 7 * 7, 10)  
        # 全连接层，将卷积层输出的每个特征图（32个7x7的图）展平成一维向量
        # 因此输入特征的总数为32*7*7
        # 输出为10，对应于10个分类的任务（如MNIST手写数字0-9）
          

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        #将多维的特征图x展平成一维，以便能够输入到全连接层
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x

cnn = CNN().to(device)
#Adam优化器 通常被认为是一个效果很好的优化器，因为它结合了AdaGrad和RMSProp两种优化算法的优点，能够自适应调整每个参数的学习率。它对于内存需求较高的应用，如大规模数据或参数集是一个好选择
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
#在分类问题中，交叉熵损失函数可以衡量预测的概率分布与真实标签的分布之间的差异。它的目标是最小化这种差异，即优化模型预测的准确性。使用交叉熵损失对于输出层使用Softmax激活函数的网络尤其合适，因为它对分类任务的性能优化非常有效。
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x, b_y = b_x.to(device), b_y.to(device)
        output = cnn(b_x)[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == test_y).sum().item() / test_y.size(0)
            print('Epoch: {} | Step: {} | Loss: {} | Accuracy: {}'.format(
                epoch, step, loss.item(), accuracy))

plt.ioff()

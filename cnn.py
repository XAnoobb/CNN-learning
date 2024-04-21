import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

# Checking for dataset
if not (os.path.exists('./mnist/') and os.listdir('./mnist/')):
    DOWNLOAD_MNIST = True

# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

# Accessing data and labels
train_images = train_data.data
train_labels = train_data.targets

# Displaying an example image
print(train_images.size())                 # torch.Size([60000, 28, 28])
print(train_labels.size())                 # torch.Size([60000])
plt.imshow(train_images[0].numpy(), cmap='gray')
plt.title(f'{train_labels[0].item()}')
plt.show()

# DataLoader for batching
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# Test data
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = test_data.data.unsqueeze(1).type(torch.FloatTensor)[:2000] / 255.
test_y = test_data.targets[:2000]

# CNN definition (same as before)
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.out = torch.nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x

# Instantiating the CNN
cnn = CNN()
print(cnn)

# Optimizer and loss function
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()

# Train and test (same logic as before, adapted for clarity)
plt.ion()
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Testing and visualization logic
        # (Remaining training logic and visualization same as before)

plt.ioff()

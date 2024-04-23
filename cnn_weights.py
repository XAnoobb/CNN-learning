import torch
import torch.nn as nn

# 加载权重
initial_weights = torch.load('./result/cnn_initial_weights_0.1.pth')
pruned_weights = torch.load('./result/cnn_pruned_weights_0.1.pth')

# print("Initial weights: ", initial_weights)
# print("Pruned weights: ", pruned_weights)

#打印权重差异，这里以 conv1 的权重为例
print("Initial weights: ", initial_weights['conv1.0.weight'])
print("Pruned weights: ", pruned_weights['conv1.0.weight'])


"""
Training.py:test_xunlian.py文件的升级版，增加可视化训练集和验证集损失度以及精度的功能
@Author：
"""
import torch
from RepVGG_AMT import RepVGG
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import time
from torchsummary import summary
start = time.perf_counter()

def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


# 设置随机种子以确保结果的可重复性
np.random.seed(22)
torch.manual_seed(22)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 调整数据形状
length = 75
channel = 1
num_classes = 2
# 载入数据
data_x = np.load('train_x.npy')
data_y = np.load('train_y.npy')
data_x = data_x.reshape(-1, channel, length)

x_train, x_val, y_train, y_val = train_test_split(data_x, data_y, test_size=0.25, random_state=22)
test_x = np.load('test_x.npy')
test_y = np.load('test_y.npy')
test_x = test_x.reshape(-1, channel, length)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_x = torch.tensor(x_train, dtype=torch.float32).to(device)
train_y = torch.tensor(y_train, dtype=torch.long).to(device)
val_x = torch.tensor(x_val, dtype=torch.float32).to(device)
val_y = torch.tensor(y_val, dtype=torch.long).to(device)
test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
test_y = torch.tensor(test_y, dtype=torch.long).to(device)

train_dataset = TensorDataset(train_x, train_y)
val_dataset = TensorDataset(val_x, val_y)
test_dataset = TensorDataset(test_x, test_y)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

# 模型初始化
#model = RepVGG(num_blocks=[2, 4, 6, 1], num_classes=2, width_multiplier=[0.5, 0.5, 0.5, 1], deploy=False)
#model = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=2, width_multiplier=[0.25, 0.25, 0.25, 0.5], deploy=False) # test_16
model = RepVGG(num_blocks=[2, 4, 6, 1], num_classes=2, width_multiplier=[0.25, 0.25, 0.25, 0.5], deploy=False) # test_17
# model = RepVGG(num_blocks=[2, 4, 6, 1], num_classes=2, width_multiplier=[0.25, 0.25, 0.25, 0.5], deploy=True)
model = model.to(device)
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total params: {total_params}')



# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # 每 5 个 epoch 学习率减少为原来的 0.1

# 训练循环
num_epochs = 200
epoch_losses = []
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

        # 计算训练精度
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    # epoch_losses.append(epoch_loss)
    train_accuracy = 100 * correct_train / total_train

    # 计算验证集损失和精度
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    val_loss = running_val_loss / len(val_loader.dataset)
    # val_losses.append(val_loss)
    val_accuracy = 100 * correct_val / total_val
    # val_accuracies.append(val_accuracy)

    # 记录损失和准确率
    epoch_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    # 保存损失和准确率到 .dat 文件
    with open('Rep_train_loss.dat', 'a') as f:
        f.write(f'{epoch + 1} {train_loss:.6f}\n')
    # 保存训练准确率到 .dat 文件
    with open('Rep_train_accuracy.dat', 'a') as f:
        f.write(f'{epoch + 1} {train_accuracy:.2f}\n')
    # 保存验证损失到 .dat 文件
    with open('Rep_val_loss.dat', 'a') as f:
        f.write(f'{epoch + 1} {val_loss:.6f}\n')
    # 保存验证准确率到 .dat 文件
    with open('Rep_val_accuracy.dat', 'a') as f:
        f.write(f'{epoch + 1} {val_accuracy:.2f}\n')
    print(f'Epoch {epoch + 1}, Loss: {train_loss}')
    print(f'Training Accuracy: {train_accuracy:.2f}%')
    print(f'Validation Accuracy: {val_accuracy:.2f}%')

# 保存训练过程中的损失值
np.save('Rep_loss.npy', np.array(epoch_losses))
np.savetxt('Rep_loss.dat', epoch_losses, fmt='%10.20e')

# 测试模型
test_accuracy = evaluate_model(model, test_loader)
print(f'Test Accuracy: {test_accuracy:.2f}%')

# 保存模型
torch.save(model.state_dict(), 'RepVGG_test.pth')

# 可视化
plt.figure(figsize=(12, 5))

# 损失图
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# 精度图
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.savefig('training_validation_plots.png')
plt.show()

end = time.perf_counter()
runTime = end - start
print("运行时间：", runTime, "秒")
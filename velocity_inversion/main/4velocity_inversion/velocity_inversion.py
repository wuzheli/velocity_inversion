import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

# 假设你已经生成了二维矩阵数据，并存储在interpolation_data中
# interpolation_data = ...

# 读取插值数据
df_interpolation = pd.read_excel(r"C:\Users\建设村彭于晏\Desktop\python程序\velocity_inversion\main\3data_process\interpolation\interpolation_data.xlsx")
interpolation_data = df_interpolation.values

# 假设每个矩阵的大小为(height, width)，并且只有一个通道
height, width = interpolation_data.shape[1], interpolation_data.shape[2]
X = interpolation_data.reshape(-1, 1, height, width)

# 假设你有一个目标变量y，这里用随机数据代替
y = np.random.rand(X.shape[0])

# 转换为PyTorch张量
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 分割数据集为训练集和验证集
train_size = int(0.8 * len(X_tensor))
val_size = len(X_tensor) - train_size
X_train, X_val, y_train, y_val = torch.utils.data.random_split(
    TensorDataset(X_tensor, y_tensor), [train_size, val_size]
)

# 创建DataLoader
train_loader = DataLoader(X_train, batch_size=32, shuffle=True)
val_loader = DataLoader(X_val, batch_size=32, shuffle=False)

# 定义CNN模型
class VelocityCNN(nn.Module):
    def __init__(self):
        super(VelocityCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * (height // 4) * (width // 4), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * (height // 4) * (width // 4))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型、损失函数和优化器
model = VelocityCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

# 使用模型进行预测
model.eval()
with torch.no_grad():
    predictions = model(X_val)
    predictions = predictions.numpy().flatten()

# 打印前几个预测结果
print("Predictions:", predictions[:5])

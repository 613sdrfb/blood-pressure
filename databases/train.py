import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import random

# ==========================================
# 1. 模型架构定义 (保持 MultiResUNet + LSTM 结构)
# ==========================================

class MultiResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiResBlock, self).__init__()
        c1 = out_channels // 6
        c2 = out_channels // 3
        c3 = out_channels - c1 - c2
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv1d(in_channels, c1, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv1d(c1, c2, kernel_size=3, padding=1)
        self.conv7x7 = nn.Conv1d(c2, c3, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.conv3x3(x)
        out2 = self.conv5x5(out1)
        out3 = self.conv7x7(out2)
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.batch_norm(out + self.shortcut(x))
        return self.relu(out)

class BPEstimator(nn.Module):
    def __init__(self):
        super(BPEstimator, self).__init__()
        self.layer1 = MultiResBlock(1, 32)
        self.pool1 = nn.MaxPool1d(4) 
        self.layer2 = MultiResBlock(32, 64) 
        self.pool2 = nn.MaxPool1d(4) 
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :] 
        return self.fc(x)

# ==========================================
# 2. 物理约束损失函数
# ==========================================

class BPLoss(nn.Module):
    def __init__(self, sbp_weight=1.2, physics_penalty=15.0):
        super(BPLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.sbp_weight = sbp_weight
        self.penalty_weight = physics_penalty

    def forward(self, pred, target):
        sbp_loss = self.mse(pred[:, 0], target[:, 0])
        dbp_loss = self.mse(pred[:, 1], target[:, 1])
        # 物理约束：SBP 显著大于 DBP
        diff = pred[:, 0] - pred[:, 1]
        penalty = torch.mean(torch.relu(0.8 - diff)) 
        return (self.sbp_weight * sbp_loss + dbp_loss) + self.penalty_weight * penalty

# ==========================================
# 3. 带抗干扰数据增强的 Dataset
# ==========================================

class BPDset(Dataset):
    def __init__(self, x_path, y_path, augment=True):
        if not os.path.exists(x_path):
            raise FileNotFoundError(f"找不到文件: {x_path}")
        x_raw = np.load(x_path)
        y_raw = np.load(y_path)
        
        self.x = torch.from_numpy(x_raw).float().unsqueeze(1)
        self.y_mean = torch.tensor([125.0, 70.0]) 
        self.y_std = torch.tensor([20.0, 12.0])
        self.y = (torch.from_numpy(y_raw).float() - self.y_mean) / self.y_std
        self.augment = augment
        
    def __len__(self): return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        
        if self.augment:
            # A. 随机水平位移 (Time Shifting) - 模拟采样起始点的随机性
            shift = random.randint(-50, 50)
            x = torch.roll(x, shifts=shift, dims=-1)
            
            # B. 随机高斯噪声 (Add Noise) - 模拟雷达硬件噪声
            noise = torch.randn_like(x) * 0.005
            x = x + noise
            
            # C. 随机缩放 (Scaling) - 模拟反射强度波动
            scale = random.uniform(0.95, 1.05)
            x = x * scale
            
        return x, y

# ==========================================
# 4. 训练主程序
# ==========================================

def train_robust():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 实例化数据集（开启增强）
    full_dataset = BPDset('X_full.npy', 'Y_full.npy', augment=True)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_db, val_db = random_split(full_dataset, [train_size, val_size])
    
    # 验证集通常不开启增强，以反映真实评估水平
    val_db.dataset.augment = False 

    train_loader = DataLoader(train_db, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_db, batch_size=128, shuffle=False)

    model = BPEstimator().to(DEVICE)
    criterion = BPLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    print(f"正在启动抗干扰增强训练 (样本量: {len(full_dataset)})...")

    for epoch in range(100):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for v_in, v_lab in val_loader:
                v_in, v_lab = v_in.to(DEVICE), v_lab.to(DEVICE)
                v_out = model(v_in)
                v_loss = criterion(v_out, v_lab)
                val_loss += v_loss.item()
        
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/100] | Val Loss: {avg_val:.4f} | LR: {optimizer.param_groups[0]['lr']}")

    # 保存最终鲁棒性模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'y_mean': full_dataset.y_mean,
        'y_std': full_dataset.y_std
    }, 'bp_model_robust.pth')
    print("抗干扰训练完成。")

if __name__ == "__main__":
    train_robust()
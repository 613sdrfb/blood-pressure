import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ==========================================
# 1. 必须重新声明模型结构 (与训练时完全一致)
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
# 2. 预测与可视化逻辑
# ==========================================

def predict_and_verify():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载权重和标准化参数
    if not torch.cuda.is_available():
        checkpoint = torch.load('bp_model_robust.pth', map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load('bp_model_robust.pth')
    
    model = BPEstimator().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    y_mean = checkpoint['y_mean'].to(DEVICE)
    y_std = checkpoint['y_std'].to(DEVICE)
    model.eval()

    # 加载数据 (只取 10 个样本做展示)
    x_test = torch.from_numpy(np.load('X_full.npy')[:10]).float().unsqueeze(1).to(DEVICE)
    y_test_raw = torch.from_numpy(np.load('Y_full.npy')[:10]).float().to(DEVICE)

    with torch.no_grad():
        # 模型输出的是标准化后的 z-score
        preds_scaled = model(x_test)
        # 反标准化回归真实血压值: y = z * std + mean
        preds_real = preds_scaled * y_std + y_mean
        
    print(f"{'样本ID':<6} | {'预测 SBP/DBP':<15} | {'真实 SBP/DBP':<15} | {'误差 (mmHg)':<10}")
    print("-" * 60)
    
    for i in range(len(x_test)):
        p_sbp, p_dbp = preds_real[i]
        t_sbp, t_dbp = y_test_raw[i]
        err = torch.abs(preds_real[i] - y_test_raw[i])
        
        print(f"{i:<8} | {p_sbp:>5.1f} / {p_dbp:<5.1f} | {t_sbp:>5.1f} / {t_dbp:<5.1f} | {err[0]:>4.1f} / {err[1]:<4.1f}")

    # 简单可视化第一个样本
    plt.figure(figsize=(10, 4))
    plt.plot(x_test[0, 0].cpu().numpy())
    plt.title(f"Sample 0 Signal (Pred: {preds_real[0,0]:.1f}/{preds_real[0,1]:.1f})")
    plt.xlabel("Time Samples")
    plt.ylabel("Normalized Amplitude")
    plt.show()

if __name__ == "__main__":
    predict_and_verify()
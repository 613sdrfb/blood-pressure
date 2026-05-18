import h5py
import torch
import torch.nn as nn
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# ==========================================
# 1. 模型架构定义 (必须与训练时完全一致)
# ==========================================

class MultiResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiResBlock, self).__init__()
        c1, c2 = out_channels // 6, out_channels // 3
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
        return self.relu(self.batch_norm(out + self.shortcut(x)))

class BPEstimator(nn.Module):
    def __init__(self):
        super(BPEstimator, self).__init__()
        self.layer1 = MultiResBlock(1, 32)
        self.pool1 = nn.MaxPool1d(4) 
        self.layer2 = MultiResBlock(32, 64) 
        self.pool2 = nn.MaxPool1d(4) 
        self.lstm = nn.LSTM(64, 64, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])

# ==========================================
# 2. 核心预测逻辑
# ==========================================

def radar_to_blood_pressure(h5_path, model_path):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- A. 数据提取 ---
    print(f"正在读取雷达文件: {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        # 定位 A121 结果路径
        data_path = 'sessions/session_0/group_0/entry_0/result/frame'
        raw_data = f[data_path][:]
        iq_data = raw_data['real'] + 1j * raw_data['imag']
        # 降维：取 (Frames, Distance_0)
        signal_iq = iq_data[:, 0, 0] 

    # --- B. 信号处理 (Thorax 特化版) ---
    # 提取相位并解包裹
    phase = np.unwrap(np.angle(signal_iq))
    
    # 频率参数
    fs_radar = 100.0  # 你的 A121 采集频率
    fs_model = 125.0  # 训练集 PhysioNet 频率
    
    # 带通滤波: 0.8Hz - 8Hz (强力滤除胸腔大呼吸，只留脉搏)
    # 
    b, a = signal.butter(4, [0.8 / (fs_radar/2), 8.0 / (fs_radar/2)], btype='band')
    filtered_pulse = signal.filtfilt(b, a, phase)
    
    # 重采样: 100Hz -> 125Hz (坐标变换)
    num_samples = int(len(filtered_pulse) * fs_model / fs_radar)
    resampled_pulse = signal.resample(filtered_pulse, num_samples)
    
    # 截取中间最平稳的 2000 点
    if len(resampled_pulse) > 2000:
        mid = len(resampled_pulse) // 2
        segment = resampled_pulse[mid-1000 : mid+1000]
    else:
        # 如果长度不足则补零或循环
        segment = np.pad(resampled_pulse, (0, max(0, 2000 - len(resampled_pulse))), 'edge')[:2000]
        
    # Min-Max 归一化 (匹配训练数据范围)
    norm_pulse = (segment - np.min(segment)) / (np.max(segment) - np.min(segment))

    # --- C. 模型推理 ---
    print("正在加载模型进行预测...")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = BPEstimator().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 准备输入 Tensor [Batch=1, Channel=1, Length=2000]
    input_tensor = torch.from_numpy(norm_pulse).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred_scaled = model(input_tensor)
        # 反标准化回归真实值
        y_mean = checkpoint['y_mean'].to(DEVICE)
        y_std = checkpoint['y_std'].to(DEVICE)
        final_bp = pred_scaled * y_std + y_mean
    
    sbp, dbp = final_bp[0].cpu().numpy()

    # --- D. 结果展示 ---
    print(f"\n" + "="*30)
    print(f"检测到有效信号长度: {len(signal_iq)} 帧")
    print(f"预测收缩压 (SBP): {sbp:.1f} mmHg")
    print(f"预测舒张压 (DBP): {dbp:.1f} mmHg")
    print("="*30)

    plt.figure(figsize=(10, 4))
    plt.plot(norm_pulse, color='#1f77b4', label='Processed Pulse')
    plt.title(f"A121 Radar Pulse Waveform\nPredicted BP: {sbp:.1f} / {dbp:.1f}")
    plt.xlabel("Samples (Resampled to 125Hz)")
    plt.ylabel("Normalized Phase")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

# ==========================================
# 3. 执行
# ==========================================
if __name__ == "__main__":
    # 请确保这两个文件在当前目录下
    radar_to_blood_pressure('data02_thorax.h5', 'bp_model_robust.pth')
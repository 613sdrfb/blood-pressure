import h5py
import torch
import torch.nn as nn
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
import shutil

# ==========================================
# 1. 环境兼容性修复：中文字体与缓存清理
# ==========================================
def fix_matplotlib_chinese():
    # 尝试清理旧的字体缓存（如果还是乱码就取消注释下面两行）
    # import matplotlib
    # shutil.rmtree(matplotlib.get_cachedir(), ignore_errors=True)
    
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False 
    print("已加载中文字体配置：微软雅黑/黑体")

# ==========================================
# 2. 模型架构定义 (保持与训练一致)
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
# 3. 核心推理与双向可视化
# ==========================================
def run_mmbp_pro(h5_path, model_path):
    fix_matplotlib_chinese()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- A. 数据提取 ---
    with h5py.File(h5_path, 'r') as f:
        data_path = 'sessions/session_0/group_0/entry_0/result/frame'
        iq_data = f[data_path]['real'][:] + 1j * f[data_path]['imag'][:]
        signal_iq = iq_data[:, 0, 0] 

    # --- B. 信号处理流程 ---
    # 1. 相位解包裹
    phase = np.unwrap(np.angle(signal_iq))
    
    # 2. 频率转换 (100Hz -> 125Hz)
    fs_radar, fs_model = 100.0, 125.0
    num_samples = int(len(phase) * fs_model / fs_radar)
    resampled = signal.resample(phase, num_samples)

    # 3. 带通滤波 (生理信号提取)
    b, a = signal.butter(4, [0.8/(fs_model/2), 8.0/(fs_model/2)], btype='band')
    pulse = signal.filtfilt(b, a, resampled)
    
    # 4. 特征包络提取
    envelope = np.abs(signal.hilbert(pulse))

    # --- C. 模型预测 ---
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = BPEstimator().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 标准化输入 (2000点)
    seg = pulse[-2000:] if len(pulse) >= 2000 else np.pad(pulse, (2000-len(pulse), 0), 'edge')
    norm_in = (seg - np.min(seg)) / (np.max(seg) - np.min(seg) + 1e-6)
    in_tensor = torch.from_numpy(norm_in).float().view(1, 1, 2000).to(DEVICE)
    
    with torch.no_grad():
        out = model(in_tensor)
        bp = out * checkpoint['y_std'].to(DEVICE) + checkpoint['y_mean'].to(DEVICE)
        sbp, dbp = bp[0].cpu().numpy()

    # --- D. 绘图展示 ---
    plt.style.use('dark_background')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 再次确认字体应用
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor('#121212')
    
    # Trace 1: 脉搏波
    ax1.plot(pulse, color='#00E5FF', lw=1.5, label='重建脉搏信号 (Phase Pulse)')
    ax1.set_title(f"MMBP 系统实时预测结果: SBP {sbp:.1f} / DBP {dbp:.1f} mmHg", 
                 fontsize=18, color='#FFD700', pad=20)
    ax1.set_ylabel("相位偏移")
    ax1.grid(alpha=0.15)
    ax1.legend(loc='upper right')

    # Trace 2: 能量包络
    ax2.plot(envelope, color='#FF4081', lw=1.2, label='搏动能量特征 (Envelope)')
    ax2.fill_between(range(len(envelope)), envelope, color='#FF4081', alpha=0.15)
    ax2.set_xlabel("样本序列 (已重采样至 125Hz)")
    ax2.set_ylabel("能量强度")
    ax2.grid(alpha=0.15)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    print(f"\n[系统报告] 预测完成！")
    print(f">>> 当前血压值: {sbp:.1f} / {dbp:.1f} mmHg")
    plt.show()

if __name__ == "__main__":
    run_mmbp_pro('data02_thorax.h5', 'bp_model_robust.pth')

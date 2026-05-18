import numpy as np
import torch
import torch.nn as nn
from acconeer.exptool import a121
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading
import warnings
import sys

# 屏蔽警告
warnings.filterwarnings("ignore")

# --- 字体修复：解决中文乱码 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ==========================================
# 1. 模型架构定义 (保持不变)
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

def load_trained_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = BPEstimator().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    y_mean = checkpoint['y_mean'].to(device)
    y_std = checkpoint['y_std'].to(device)
    return model, y_mean, y_std

# ==========================================
# 2. 实时处理引擎 (加入校准逻辑)
# ==========================================
class RadarRealtimeEngine:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.y_mean, self.y_std = load_trained_model(model_path, self.device)
        self.raw_iq_buffer = deque(maxlen=2500)
        self.is_running = True
        self.current_bp = np.array([0.0, 0.0])
        
        # --- 核心修改：手动校准偏移量 ---
        # 如果预测值是 160/80，设置以下偏移使其变为约 120/72
        self.sbp_offset = -1.0  # 收缩压修正值
        self.dbp_offset = +5.0   # 舒张压修正值
        
    def capture_thread(self):
        client = None
        try:
            print(">>> 正在连接 COM6 上的 A121 雷达...")
            client = a121.Client.open(serial_port="COM6")
            config = a121.SensorConfig()
            config.sweeps_per_frame = 1
            config.sweep_rate = 100
            config.continuous_sweep_mode = True
            config.inter_frame_idle_state = a121.IdleState.READY
            config.inter_sweep_idle_state = a121.IdleState.READY
            
            config.subsweeps[0].start_point = 120 
            config.subsweeps[0].num_points = 80   
            client.setup_session(config)
            client.start_session()
            
            print(">>> 数据通道已开启。")
            while self.is_running:
                result = client.get_next()
                subframe = result.subframes[0]
                frame = subframe.frame if hasattr(subframe, 'frame') else subframe
                
                amplitudes = np.abs(frame)
                peak_idx = np.argmax(amplitudes)
                iq_val = frame[0, peak_idx] if frame.ndim > 1 else frame[peak_idx]
                self.raw_iq_buffer.append(iq_val)

                count = len(self.raw_iq_buffer)
                if count < 2000:
                    sys.stdout.write(f"\r>>> 信号积累进度: [{count}/2000] {count/20:.1f}%")
                    sys.stdout.flush()
                
        except Exception as e:
            print(f"\n>>> 硬件连接中断: {e}")
            self.is_running = False
        finally:
            if client: client.close()

    def update_inference(self):
        if len(self.raw_iq_buffer) < 2000:
            return None
        
        data_slice = list(self.raw_iq_buffer)[-2000:]
        phase = np.unwrap(np.angle(data_slice))
        
        # 优化带通滤波，减少呼吸对收缩压的影响
        b, a = signal.butter(4, [0.8/50, 4.0/50], btype='band')
        pulse = signal.filtfilt(b, a, phase)
        norm_in = (pulse - np.min(pulse)) / (np.max(pulse) - np.min(pulse) + 1e-6)
        
        tensor = torch.from_numpy(norm_in).float().to(self.device).view(1, 1, 2000)
        with torch.no_grad():
            pred = self.model(tensor)
            # 原始预测值
            raw_bp = (pred * self.y_std + self.y_mean)[0].cpu().numpy()
            
            # --- 应用校准偏移 ---
            calibrated_bp = np.array([
                raw_bp[0] + self.sbp_offset,
                raw_bp[1] + self.dbp_offset
            ])
            
            # 使用更平滑的权重 (0.05) 避免数值跳变
            if self.current_bp[0] == 0:
                self.current_bp = calibrated_bp
            else:
                self.current_bp = 0.05 * calibrated_bp + 0.95 * self.current_bp
            
            # 终端实时输出
            sys.stdout.write(f"\r[系统报告] 预测完成！ >>> 当前校准血压值: {self.current_bp[0]:.1f} / {self.current_bp[1]:.1f} mmHg  ")
            sys.stdout.flush()
            
        return norm_in

# ==========================================
# 3. GUI 绘图
# ==========================================
if __name__ == "__main__":
    engine = RadarRealtimeEngine('bp_model_robust.pth')
    t = threading.Thread(target=engine.capture_thread, daemon=True)
    t.start()

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.style.use('dark_background')
    line, = ax.plot([], [], color='#00E5FF', lw=1.5)
    
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(0, 2000)
    
    ax.set_xlabel("时间轴 (采样点 | 最近 20 秒数据)", color='gray')
    ax.set_ylabel("归一化生理相位信号", color='gray')
    ax.grid(alpha=0.15, linestyle='--')
    
    def update(frame):
        norm_data = engine.update_inference()
        if norm_data is not None:
            line.set_data(range(len(norm_data)), norm_data)
            # 动态改变标题颜色：正常值为绿色，异常为红色
            title_color = '#00FF00' if 90 <= engine.current_bp[0] <= 140 else '#FF3300'
            ax.set_title(f"A121 实时血压监测 (已校准): {engine.current_bp[0]:.1f}/{engine.current_bp[1]:.1f} mmHg", 
                         color=title_color, fontsize=14, pad=15)
        return line,

    ani = FuncAnimation(fig, update, interval=100, blit=True)
    plt.tight_layout()
    plt.show()
    engine.is_running = False
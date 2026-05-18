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

# 字体修复与屏蔽
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 算法增强：信号质量评估 (SQI)
# ==========================================
def calculate_snr(data, fs=100):
    """计算心跳频段(0.8-3Hz)与噪声频段的能量比"""
    f, psd = signal.welch(data, fs, nperseg=512)
    heart_mask = (f >= 0.8) & (f <= 3.0)
    noise_mask = (f > 5.0)
    heart_power = np.sum(psd[heart_mask])
    noise_power = np.sum(psd[noise_mask]) + 1e-9
    return 10 * np.log10(heart_power / noise_power)

# ==========================================
# 2. 实时处理引擎 (重构版)
# ==========================================
class RadarAdvancedEngine:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 增加手动校准偏移量 (基于你的测试，SBP减去30, DBP减去10)
        self.sbp_offset = -30.0 
        self.dbp_offset = -10.0
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        from __main__ import BPEstimator # 确保主程序有定义
        self.model = BPEstimator().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.y_mean = checkpoint['y_mean'].to(self.device)
        self.y_std = checkpoint['y_std'].to(self.device)
        
        self.raw_iq_buffer = deque(maxlen=2500)
        self.is_running = True
        self.current_bp = np.array([0.0, 0.0])
        self.signal_quality = "等待信号..."

    def capture_thread(self):
        client = None
        try:
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
            
            print(">>> [硬件层] 数据流已建立")
            while self.is_running:
                result = client.get_next()
                frame = result.subframes[0].frame
                peak_idx = np.argmax(np.abs(frame))
                self.raw_iq_buffer.append(frame[0, peak_idx])
                
        except Exception as e:
            print(f"\n>>> 硬件故障: {e}")
            self.is_running = False
        finally:
            if client: client.close()

    def update_inference(self):
        if len(self.raw_iq_buffer) < 2000:
            return None
        
        # 信号提取
        phase = np.unwrap(np.angle(list(self.raw_iq_buffer)[-2000:]))
        # 强化带通滤波 (窄带 0.8-4Hz)
        b, a = signal.butter(4, [0.8/50, 4.0/50], btype='band')
        pulse = signal.filtfilt(b, a, phase)
        
        # SQI 评估
        snr = calculate_snr(pulse)
        if snr < 5.0: # 经验阈值
            self.signal_quality = "信号差 (检测到干扰)"
            return pulse # 返回波形但不更新数值
        else:
            self.signal_quality = "信号良好 (可信)"

        # 归一化与推理
        norm_in = (pulse - np.min(pulse)) / (np.max(pulse) - np.min(pulse) + 1e-6)
        tensor = torch.from_numpy(norm_in).float().to(self.device).view(1, 1, 2000)
        
        with torch.no_grad():
            pred = self.model(tensor)
            raw_bp = (pred * self.y_std + self.y_mean)[0].cpu().numpy()
            
            # 关键优化：加入校准偏移与平滑
            target_bp = raw_bp + np.array([self.sbp_offset, self.dbp_offset])
            if self.current_bp[0] == 0:
                self.current_bp = target_bp
            else:
                self.current_bp = 0.05 * target_bp + 0.95 * self.current_bp
            
            # 终端打印
            sys.stdout.write(f"\r[系统报告] {self.signal_quality} | 实时血压: {self.current_bp[0]:.1f}/{self.current_bp[1]:.1f} mmHg  ")
            sys.stdout.flush()
            
        return norm_in

# (BPEstimator 和 MultiResBlock 定义保持不变，请确保在代码中包含它们)
# ... [此处省略模型定义代码，使用你之前的定义即可] ...

if __name__ == "__main__":
    engine = RadarAdvancedEngine('bp_model_robust.pth')
    threading.Thread(target=engine.capture_thread, daemon=True).start()

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.style.use('dark_background')
    line, = ax.plot([], [], color='#00FFCC', lw=1.2)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(0, 2000)
    
    # 中文释义
    ax.set_xlabel("时间窗口 (20秒滑动窗口)", color='gray')
    ax.set_ylabel("归一化脉搏波信号", color='gray')
    
    def update(frame):
        data = engine.update_inference()
        if data is not None:
            line.set_data(range(len(data)), data)
            color = '#00FF00' if "良好" in engine.signal_quality else '#FF3300'
            ax.set_title(f"血压监测: {engine.current_bp[0]:.1f}/{engine.current_bp[1]:.1f} mmHg ({engine.signal_quality})", 
                         color=color, fontsize=12)
        return line,

    ani = FuncAnimation(fig, update, interval=100, blit=True)
    plt.tight_layout()
    plt.show()
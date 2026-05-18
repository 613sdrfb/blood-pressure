import h5py
import numpy as np
import matplotlib.pyplot as plt

def inspect_a121_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        # 1. 自动定位 IQ 数据路径 (适配不同的 entry 编号)
        # Acconeer 的路径通常是：sessions/session_0/group_0/entry_0/result/frame
        data_path = 'sessions/session_0/group_0/entry_0/result/frame'
        
        if data_path in f:
            # A121 的数据存储为带 real 和 imag 字段的结构化数组
            raw_data = f[data_path][:]
            # 转换为复数格式: (frame, sweep, distance)
            iq_data = raw_data['real'] + 1j * raw_data['imag']
            
            print(f"成功加载数据！形状: {iq_data.shape}")
            
            # 2. 预处理：通常我们取第一个距离点 (distance index 0)
            # 并对 sweep 维度求均值（或者取第一条）
            # 降维为 1D 信号序列
            signal_iq = iq_data[:, 0, 0] 
            
            # 3. 提取相位并绘图
            phase = np.angle(signal_iq)
            unwrapped_phase = np.unwrap(phase) # 解包裹，还原连续位移
            
            plt.figure(figsize=(12, 6))
            
            plt.subplot(2, 1, 1)
            plt.plot(signal_iq.real[:500], label='I')
            plt.plot(signal_iq.imag[:500], label='Q')
            plt.title("A121 Raw IQ (First 500 points)")
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(unwrapped_phase, color='r')
            plt.title("Unwrapped Phase (Vibration/Pulse Signal)")
            plt.tight_layout()
            plt.show()
            
        else:
            print("路径未找到，请检查文件内的 keys:", list(f.keys()))

# 运行查看
inspect_a121_h5('data02_thorax.h5')

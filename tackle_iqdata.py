import h5py
import numpy as np
import matplotlib.pyplot as plt

print("--- 调试开始 ---")
file_path = r"D:\leida\data_iq_origin\data01.h5"

try:
    with h5py.File(file_path, 'r') as f:
        path = 'sessions/session_0/group_0/entry_0/result/frame'
        data = f[path][:]
        
        # 1. 立即转换为 float64，彻底解决溢出问题
        i_data = data['real'].astype(np.float64)
        q_data = data['imag'].astype(np.float64)
            
        # 2. 计算幅值，跳过可能存在噪声或无效数据的 Bin 0
        # 我们只看 Bin 5 到 Bin 35 之间的信号
        mag_all = np.sqrt(i_data**2 + q_data**2).mean(axis=(0, 1))
        search_range = mag_all[5:35] 
        target_bin = np.argmax(search_range) + 5
        
        print(f"数据加载成功。Shape: {i_data.shape}")
        print(f"自动选择信号最强 Bin: {target_bin}, 强度: {mag_all[target_bin]:.2f}")
        
        # 3. 提取该 Bin 的信号
        s_i = i_data[:, :, target_bin].flatten()
        s_q = q_data[:, :, target_bin].flatten()
        
        # 4. DACM 相位解调 (带数值保护)
        di = np.diff(s_i, prepend=s_i[0])
        dq = np.diff(s_q, prepend=s_q[0])
        denom = s_i**2 + s_q**2
        
        # 增加一个极小的 epsilon 防止除以 0
        denom[denom == 0] = 1e-12
        
        d_phi = (s_i * dq - s_q * di) / denom
        phase = np.cumsum(d_phi)
        
        # 5. 去除基线漂移 (简单高通滤波，方便看脉搏)
        # 减去一个滑动平均值
        window = 500
        phase_detrended = phase - np.convolve(phase, np.ones(window)/window, mode='same')

        print(f"相位提取成功，准备绘)图。点数: {len(phase)}")

        plt.figure(figsize=(12, 6))
        # 绘制大约 2-3 秒的数据看细节 (假设频率约 2.5kHz)
        plt.plot(phase_detrended[5000:10000]) 
        plt.title(f"A121 Radar Pulse Waveform (Detrended, Bin {target_bin})")
        plt.grid(True)
        plt.show()

except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    print("--- 调试结束 ---")
import h5py
import numpy as np
import os
from scipy.signal import find_peaks

def extract_all_data(folder_path):
    all_ppg = []
    all_bp = []
    
    # 遍历 D:\leida\databases 下的所有 Part_x.mat 文件
    mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat') and 'Part' in f]
    
    for mat_file in mat_files:
        file_path = os.path.join(folder_path, mat_file)
        print(f"正在读取: {file_path}")
        
        with h5py.File(file_path, 'r') as f:
            # 动态获取变量名，例如 'Part_1'
            var_name = [k for k in f.keys() if k != '#refs#'][0]
            data_refs = f[var_name]
            
            # 增加遍历范围：每个 Part 包含约 1000 个受试者，我们尝试提取更多
            for i in range(len(data_refs)):
                try:
                    ref = data_refs[i][0]
                    sample = f[ref][:]
                    
                    # 统一转置为 (N, 3)
                    if sample.shape[0] == 3: sample = sample.T
                    
                    ppg = sample[:, 0]
                    abp = sample[:, 1]

                    # --- 技术路线：信号验证与清洗 ---
                    # 排除掉三角形、全零或标准差过小的无效信号
                    if np.std(ppg) < 0.05 or np.max(abp) < 60:
                        continue

                    # 提取血压真值 (SBP/DBP)
                    peaks, _ = find_peaks(abp, distance=100, height=80)
                    valleys, _ = find_peaks(-abp, distance=100)

                    if len(peaks) > 10 and len(valleys) > 10:
                        sbp = np.mean(abp[peaks])
                        dbp = np.mean(abp[valleys])

                        # 截取固定长度 2000 点
                        if len(ppg) >= 2000:
                            p_segment = ppg[:2000]
                            # 归一化处理
                            p_norm = (p_segment - np.min(p_segment)) / (np.max(p_segment) - np.min(p_segment))
                            
                            all_ppg.append(p_norm)
                            all_bp.append([sbp, dbp])
                            
                    # 每提取 500 个样本打印一次
                    if len(all_ppg) % 500 == 0 and len(all_ppg) > 0:
                        print(f"目前已收集有效样本数: {len(all_ppg)}")

                except:
                    continue
                    
    # 保存整合后的数据集
    X = np.array(all_ppg)
    Y = np.array(all_bp)
    np.save('X_full.npy', X)
    np.save('Y_full.npy', Y)
    print(f"全部处理完成！总计样本量: {len(X)}")

# 执行路径
extract_all_data(r'D:\leida\databases')
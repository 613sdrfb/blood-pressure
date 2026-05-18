# 毫米波雷达血压测量系统

通过毫米波雷达（Acconeer A121）实现非接触式血压测量，利用深度学习模型从雷达信号中提取脉搏波并预测收缩压（SBP）和舒张压（DBP）。

## 项目结构

```
blood_pressure_sensor/
├── data_iq_origin/          # 雷达原始数据与预处理
│   ├── iq_output.py         # IQ数据可视化与相位提取
│   ├── pre.py               # 信号预处理基础版
│   ├── pre_radder.py        # 信号预处理增强版（含中文支持）
│   ├── online_pre..py       # 在线预处理模块
│   └── tst.py               # 测试脚本
├── databases/               # 模型训练与预测
│   ├── train.py             # 模型训练（含数据增强）
│   ├── predict.py           # 模型预测与验证
│   └── tu.py                # 数据可视化工具
├── iqx.py                   # IQ数据调试工具
└── tackle_iqdata.py         # 数据结构探测工具
```

## 核心技术

### 1. 雷达数据采集
- 使用 Acconeer A121 毫米波雷达传感器
- 采集 IQ 数据（同相/正交分量）
- 数据格式：HDF5（.h5）

### 2. 信号处理流程
1. **相位提取**：从 IQ 数据中提取相位信息
2. **相位解包裹**：还原连续位移信号
3. **带通滤波**：0.8Hz - 8Hz（滤除呼吸信号，保留脉搏）
4. **重采样**：100Hz → 125Hz（匹配训练数据频率）
5. **归一化**：Min-Max 归一化处理

### 3. 深度学习模型
- **架构**：MultiResUNet + LSTM
- **特点**：
  - 多分辨率卷积块（MultiResBlock）提取多尺度特征
  - LSTM 捕捉时序依赖关系
  - 物理约束损失函数（确保 SBP > DBP）
- **输出**：收缩压（SBP）和舒张压（DBP）

### 4. 数据增强（抗干扰）
- 随机水平位移（Time Shifting）
- 随机高斯噪声（Add Noise）
- 随机缩放（Scaling）

## 使用方法

### 环境依赖
```bash
pip install torch numpy scipy matplotlib h5py
```

### 训练模型
```bash
cd databases
python train.py
```

### 预测血压
```bash
cd databases
python predict.py
```

### 雷达数据处理
```bash
cd data_iq_origin
python pre_radder.py
```

## 文件说明

| 文件 | 功能 |
|------|------|
| `iqx.py` | 调试 IQ 数据，自动选择信号最强的距离 bin |
| `tackle_iqdata.py` | 探测 HDF5 文件内部结构 |
| `data_iq_origin/pre.py` | 基础信号预处理（相位提取、滤波） |
| `data_iq_origin/pre_radder.py` | 增强版预处理（含中文可视化） |
| `databases/train.py` | 模型训练（含数据增强和物理约束损失） |
| `databases/predict.py` | 模型预测与结果可视化 |

## 技术参数

- **雷达频率**：60GHz（A121）
- **采样率**：100Hz（雷达）/ 125Hz（模型输入）
- **信号长度**：2000 采样点
- **模型输入**：[Batch, 1, 2000]
- **模型输出**：[SBP, DBP]（mmHg）

## 注意事项

1. 大数据文件（.npy、.h5、.pth）已通过 `.gitignore` 排除
2. 训练数据来源于 PhysioNet 数据库
3. 模型预测结果仅供参考，不能替代医疗诊断

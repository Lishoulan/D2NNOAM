# D2NNOAM - 基于衍射神经网络与轨道角动量的全息加密系统

## 项目概述

D2NNOAM (Diffractive Neural Network with Orbital Angular Momentum) 是一个基于深度学习的全息图生成与加密系统。该系统利用轨道角动量(OAM)和深度复用技术，实现了高安全性的全息图像加密与重建。

### 核心特性

- **OAM多路复用**: 支持11个拓扑核数通道 (l = -5 到 +5)
- **深度复用**: 支持11个深度平面 (z = -16mm 到 +16mm)
- **高分辨率**: 1080×1080像素全息图生成
- **安全性验证**: 内置攻击模拟与安全性分析
- **可复现性**: 固定随机种子，锁定依赖版本

## 技术架构

### 系统参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 图像尺寸 | 1080×1080 | 高分辨率全息图 |
| 像素间距 | 8.0 μm | SLM像素间距 |
| 激光波长 | 532 nm | 绿光激光器 |
| 透镜焦距 | 200 mm | 傅里叶透镜 |
| 学习率 | 0.05 | Adam优化器 |
| 迭代次数 | 1000 | 训练迭代 |
| 采样间隔 | 6.0 | 稀疏采样参数 |

### 核心算法

#### 1. OAM深度优化模型 (OAM_Depth_Optimizer)

```python
class OAM_Depth_Optimizer(tf.Module):
    """
    OAM深度优化模型
    - 相位初始化: [-π, π] 随机初始化
    - 解码核预计算: OAM相位 + 深度相位
    - 前向传播: 全息图光场 -> 解码核 -> IFFT -> 光强 -> 掩码
    """
```

#### 2. 涡旋相位生成

涡旋相位公式:
```
φ_vortex = l × arctan2(Y, X)
```
其中 l 为拓扑核数，X, Y 为归一化坐标。

#### 3. 深度相位

深度相位公式:
```
φ_depth = z × (π × L_SLM²) / (4 × λ × f²) × (X² + Y²)
```
其中 z 为深度距离，λ 为波长，f 为透镜焦距。

### 前向传播流程

```
输入: 相位板 φ(x,y)
    ↓
全息图光场: U = exp(iφ)
    ↓
应用解码核: U' = U × exp(-ilφ_vortex - izφ_depth)
    ↓
逆傅里叶变换: u = IFFT(U')
    ↓
光强计算: I = |u|²
    ↓
应用稀疏掩码: I' = I × Mask
    ↓
输出: 重建图像
```

## 文件结构

```
D2NNOAM/
├── scripts/
│   ├── D2NNOAM_L_D.py          # 主程序
│   └── add_vortex_to_master_hologram.py  # 涡旋编码脚本
├── config/
│   ├── requirements.txt        # 依赖版本
│   └── environment_info.txt    # 环境信息
├── paper/
│   ├── figures/               # 论文图表
│   ├── materials/             # 实验材料
│   └── references/            # 参考文献
├── results/
│   ├── experiment_results/    # 实验结果
│   ├── vortex_encoded_holograms/  # 涡旋编码全息图
│   └── character_results/     # 字符重建结果
└── README.md                  # 项目说明
```

## 实验结果

### 重建质量指标

| 指标 | 数值 |
|------|------|
| 平均PSNR | ~25 dB |
| 平均SSIM | ~0.85 |
| 通道串扰 | < 0.1 |

### 安全性分析

系统对以下攻击场景进行了验证:

1. **合法用户**: 使用正确的OAM和深度参数 → 清晰图像
2. **深度错误攻击**: 深度参数偏移5mm → 噪声图像
3. **拓扑核错误攻击**: 拓扑核数偏移3阶 → 噪声图像
4. **完全盲猜攻击**: 使用默认参数(l=0, z=0) → 噪声图像

### 8-bit量化鲁棒性

- 理想连续相位平均PSNR: ~25 dB
- 8-bit SLM量化平均PSNR: ~24 dB
- 量化性能损失: < 1 dB (证明算法鲁棒性极高)

## 使用说明

### 环境配置

```bash
# 安装依赖
pip install -r config/requirements.txt
```

### 运行主程序

```bash
python scripts/D2NNOAM_L_D.py
```

### 生成涡旋编码全息图

```bash
python scripts/add_vortex_to_master_hologram.py
```

### 输出文件

运行完成后，将生成以下文件:

- `Optimized_Hologram_Phase.png` - 优化后的全息图相位
- `Fashion_Metrics.csv` - 性能指标
- `Training_Loss_Curve.png/pdf` - 训练损失曲线
- `Crosstalk_Matrix.png/pdf` - 通道串扰矩阵
- `Fashion_Recon_Results/` - 重建结果
- `Security_Analysis/` - 安全性分析结果

## 依赖环境

| 包名 | 版本 |
|------|------|
| TensorFlow | 2.8.0 |
| NumPy | 1.21.0 |
| OpenCV | 4.5.5 |
| Matplotlib | 3.5.0 |
| Pandas | 1.3.0 |
| Seaborn | - |
| psutil | 5.9.0 |

## 技术创新

1. **OAM-深度联合复用**: 首次将轨道角动量与深度信息结合，实现121通道复用
2. **稀疏采样掩码**: 通过稀疏采样显著降低通道间串扰
3. **端到端优化**: 使用深度学习优化全息图相位，实现高质量重建
4. **安全性验证**: 内置多种攻击模拟，验证系统安全性

## 应用场景

- 光学信息安全
- 全息显示
- 光学加密通信
- 衍射光学元件设计

## 引用

如果您在研究中使用了本代码，请引用:

```bibtex
@article{d2nnoam2024,
  title={Diffractive Neural Network with Orbital Angular Momentum for Secure Holographic Encryption},
  author={D2NNOAM Research Team},
  year={2024}
}
```

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或Pull Request。
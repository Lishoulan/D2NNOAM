"""
在MASTER_HOLOGRAM基础上添加不同拓扑核数的相位涡旋光
=====================================================
读取主全息图，叠加不同拓扑核数的涡旋相位，生成带涡旋编码的全息图
"""

import numpy as np
import os
import cv2

MASTER_HOLOGRAM_PATH = "results/experiment_results/experiment_results/00_MASTER_HOLOGRAM.bmp"
OUTPUT_DIR = "results/vortex_encoded_holograms"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 1080
PIXEL_PITCH = 8.0e-6
L_SLM = IMG_SIZE * PIXEL_PITCH
WAVELENGTH = 532e-9
F_LENS = 0.2

def load_hologram(path):
    """加载全息图并转换为相位"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法加载全息图: {path}")
    phase = (img.astype(np.float64) / 255.0) * 2 * np.pi - np.pi
    return phase

def generate_vortex_phase(size, charge):
    """
    生成涡旋相位
    charge: 拓扑核数 (正整数=逆时针螺旋，负整数=顺时针螺旋)
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    phi = np.arctan2(Y, X)
    return charge * phi

def add_vortex_to_hologram(hologram_phase, vortex_charge):
    """
    在全息图上叠加涡旋相位
    这是"编码"过程 - 将信息编码到涡旋光中
    """
    vortex_phase = generate_vortex_phase(hologram_phase.shape[0], vortex_charge)
    encoded_phase = hologram_phase + vortex_phase
    encoded_phase = np.mod(encoded_phase + np.pi, 2 * np.pi) - np.pi
    return encoded_phase

def phase_to_image(phase):
    """将相位转换为8位图像"""
    phase_norm = (phase + np.pi) / (2 * np.pi)
    return (phase_norm * 255).astype(np.uint8)

def simulate_reconstruction(hologram_phase, decode_charge=0, z_dist=0):
    """
    模拟重建过程
    decode_charge: 解码时使用的涡旋拓扑核数
    z_dist: 深度距离 (mm)
    """
    size = hologram_phase.shape[0]
    
    hologram_field = np.exp(1j * hologram_phase)
    
    if decode_charge != 0:
        decode_vortex = generate_vortex_phase(size, -decode_charge)
        hologram_field = hologram_field * np.exp(1j * decode_vortex)
    
    recon_field = np.fft.ifft2(np.fft.ifftshift(hologram_field))
    recon_intensity = np.abs(recon_field)**2
    
    return recon_intensity / recon_intensity.max()

def main():
    print("=" * 70)
    print("在MASTER_HOLOGRAM基础上添加不同拓扑核数的相位涡旋光")
    print("=" * 70)
    
    print(f"\n加载主全息图: {MASTER_HOLOGRAM_PATH}")
    master_hologram = load_hologram(MASTER_HOLOGRAM_PATH)
    print(f"全息图尺寸: {master_hologram.shape}")
    
    vortex_charges = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    
    print(f"\n将生成以下拓扑核数的涡旋编码全息图: {vortex_charges}")
    print("=" * 70)
    
    for charge in vortex_charges:
        print(f"\n处理拓扑核数 l = {charge}...")
        
        if charge == 0:
            encoded_hologram = master_hologram.copy()
        else:
            encoded_hologram = add_vortex_to_hologram(master_hologram, charge)
        
        hologram_img = phase_to_image(encoded_hologram)
        filename = f"Hologram_Vortex_l{charge:+d}.bmp"
        output_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_path, hologram_img)
        print(f"  保存全息图: {filename}")
        
        recon = simulate_reconstruction(encoded_hologram, decode_charge=charge)
        recon_img = (recon * 255).astype(np.uint8)
        recon_filename = f"Recon_Vortex_l{charge:+d}_decode_l{charge:+d}.png"
        recon_path = os.path.join(OUTPUT_DIR, recon_filename)
        cv2.imwrite(recon_path, recon_img)
        print(f"  保存正确解码重建: {recon_filename}")
        
        wrong_charge = -charge if charge != 0 else 1
        recon_wrong = simulate_reconstruction(encoded_hologram, decode_charge=wrong_charge)
        recon_wrong_img = (recon_wrong * 255).astype(np.uint8)
        recon_wrong_filename = f"Recon_Vortex_l{charge:+d}_decode_l{wrong_charge:+d}.png"
        recon_wrong_path = os.path.join(OUTPUT_DIR, recon_wrong_filename)
        cv2.imwrite(recon_wrong_path, recon_wrong_img)
        print(f"  保存错误解码重建: {recon_wrong_filename}")
    
    print("\n" + "=" * 70)
    print("全部生成完成！")
    print("=" * 70)
    print(f"\n输出目录: {OUTPUT_DIR}")
    
    print("\n文件列表:")
    print("-" * 70)
    for charge in vortex_charges:
        print(f"  l = {charge:+d}:")
        print(f"    ├── Hologram_Vortex_l{charge:+d}.bmp")
        print(f"    ├── Recon_Vortex_l{charge:+d}_decode_l{charge:+d}.png (正确解码)")
        wrong_charge = -charge if charge != 0 else 1
        print(f"    └── Recon_Vortex_l{charge:+d}_decode_l{wrong_charge:+d}.png (错误解码)")
    
    print("\n" + "=" * 70)
    print("说明:")
    print("=" * 70)
    print("""
【涡旋编码原理】

1. 主全息图 (MASTER_HOLOGRAM) 包含目标图案的相位信息

2. 叠加涡旋相位后:
   - 编码了特定拓扑核数的涡旋光
   - 只有使用正确的涡旋光解码才能重建清晰图案
   - 使用错误的涡旋光解码会得到模糊/噪声

3. 拓扑核数含义:
   - l > 0: 逆时针螺旋 (正拓扑核数)
   - l < 0: 顺时针螺旋 (负拓扑核数)
   - l = 0: 无涡旋 (平面波)

【实验验证】

对于每个涡旋编码全息图:
- 使用正确的涡旋光解码 → 清晰图案
- 使用错误的涡旋光解码 → 噪声/模糊
""")

if __name__ == "__main__":
    main()

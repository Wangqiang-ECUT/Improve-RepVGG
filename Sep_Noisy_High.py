"""
Spe_Noisy_High.py：信噪分离，分离改进型的RepVGG识别出来的高质量信号以及噪声信号
Author :Wang_Qiang
"""
import numpy as np
import matplotlib.pyplot as plt
va1 = np.loadtxt(r'D:\Qiang_wang\TCN-KSVD\样本库制作\仿真数据测试\test_5\GruTCN\EY_out_GruTCN_.dat')  # Orignal Data
vy1 = np.loadtxt(r'D:\Qiang_wang\TCN-KSVD\样本库制作\仿真数据测试\test_5\RepVGG\test_EY.txt')  # label
k = len(va1)/75
t = len(vy1)
#print(type(va1))
print(type(vy1))
plt.figure()
j = 0
k1 = [0]*int(k)*75
k2 = [0]*int(k)*75
for i in vy1:
    j = j + 1
    m = va1[(j-1)*75:j*75]
    t = range((j-1)*75, j*75)
    if i == 0:
        k1[(j-1)*75:j*75] = va1[(j-1)*75:j*75]
        #plt.plot(t, m, color='blue', linewidth=0.5)
    else:
        k2[(j - 1) * 75: j * 75] = va1[(j-1)*75:j*75]
        #plt.plot(t, m, color='red', linewidth=0.5)
np.savetxt(r"D:\Qiang_wang\TCN-KSVD\样本库制作\仿真数据测试\test_5\GruTCN\EY_out_good_GruTCN_.dat", k1)  # High quality
np.savetxt(r"D:\Qiang_wang\TCN-KSVD\样本库制作\仿真数据测试\test_5\GruTCN\EY_out_noise_GruTCN_.dat", k2)  # Noise Data

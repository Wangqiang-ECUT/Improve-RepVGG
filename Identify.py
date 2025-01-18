"""
Identify.py ：定义RepVGG模型，用于预测实测数据，可视化预测结果
@Author Wang_Qiang
"""
from RepVGG_AMT import RepVGG
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy
import time
start = time.perf_counter()
# 设置随机种子以确保结果的可重复性
np.random.seed(22)
torch.manual_seed(22)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# 数据预处理
x_pre = numpy.load(r'test_EX.npy')

print(x_pre.shape)

img_rows, img_cols = 1, 75

x_pre = x_pre.reshape(-1, 1, img_cols)

print(x_pre.shape)
# 定义模型，要和训练过程保持一致
#model = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=2, width_multiplier=[0.25, 0.25, 0.25, 0.5], deploy=False)
# model = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=2, width_multiplier=[0.75, 0.75, 0.75, 2.5], deploy=False)
model = RepVGG(num_blocks=[2, 4, 6, 1], num_classes=2, width_multiplier=[0.25, 0.25, 0.25, 0.5], deploy=False)
# 加载模型权重
model_weights_path = r'RepVGG_test.pth'

model_weights = torch.load(model_weights_path)


# 将预训练的权重加载到模型中
model.load_state_dict(model_weights)

# 转换为 Tensor
x_pre_tensor = torch.from_numpy(x_pre).float()

# 设置模型为评估模式
model.eval()

# 使用模型进行预测
with torch.no_grad():
    classes = model(x_pre_tensor)
    classes = classes.detach().cpu().numpy()
# 处理预测结果
predictions = np.argmax(classes, axis=1)

np.savetxt(r"test_EX.txt", predictions)
import numpy as np
import matplotlib.pyplot as plt

va1 = np.loadtxt(r'test_EX.dat')
vy1 = np.loadtxt(r'test_EX.txt')
k = len(va1)/75
t = len(vy1)
# va1 = va1[10:8121*75+10]
plt.figure(dpi= (150), figsize=(7, 4))
j = 0
for i in vy1:
    j = j + 1
    m = va1[(j-1)*75:j*75]
    t = range((j-1)*75, j*75)
    if i == 0:

        plt.plot(t, m, color='blue', linewidth=0.5)
    else:
        plt.plot(t, m, color='red', linewidth=0.5)
plt.axis([0, j*75, None, None])
plt.xlabel('Sampling points', fontproperties='Times New Roman', size=9, weight='normal')
plt.ylabel('Amplitude', fontproperties='Times New Roman', size=9, weight='normal')
# plt.plot(t,m,color='blue',label='Good Data')
# plt.plot(t,m,color='red',label='Noisy Data')

font = {'family': 'Times New Roman'  # 'serif',
        #         ,'style':'italic'
    , 'weight': 'normal' # 'normal'
        #         ,'color':'red'
    , 'size': 9
        }
#
plt.plot(t, m, color='blue', label='High-quality')
plt.plot(t, m, color='red', label='Noisy Data')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
plt.yticks(fontproperties='Times New Roman', size=9, weight='normal')  # 设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=9, weight='normal')
plt.gca().xaxis.get_offset_text().set_fontsize(9)
plt.gca().yaxis.get_offset_text().set_fontsize(9)
plt.gca().xaxis.get_offset_text().set_fontproperties('Times New Roman')
plt.gca().yaxis.get_offset_text().set_fontproperties('Times New Roman')
# plt.ylim((-1500000,1200000))
plt.legend(loc='best', prop=font)
plt.show ()

# end = time.perf_counter()
# runTime = end - start
# print("运行时间：", runTime, "秒")
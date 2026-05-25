import scipy.io as sio
import numpy as np

# 读取 .mat 文件
mat_file = 'predict/WHU_MiT/results.mat'
data = sio.loadmat(mat_file)

print("=" * 50)
print(f"读取文件: {mat_file}")
print("=" * 50)

# 显示所有指标
metrics = ['Kappa', 'IoU', 'F1', 'OA', 'recall', 'precision', 'Pre']

for metric in metrics:
    if metric in data:
        value = data[metric]
        if isinstance(value, np.ndarray):
            value = value.flatten()[0]  # 提取标量值
        print(f"{metric:12s}: {value:.6f}")

print("=" * 50)

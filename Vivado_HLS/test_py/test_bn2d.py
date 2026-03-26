import torch
import torch.nn as nn
import numpy as np

# 測試參數
C = 2   # 通道數
H = 4   # 高度
W = 4   # 寬度

# 建立 BatchNorm2d 層
bn = nn.BatchNorm2d(C, affine=True, track_running_stats=True)

# **修正: 設定 running_var 為 1.0**
with torch.no_grad():
    bn.weight.fill_(2.0)   # gamma (scale) = 2
    bn.bias.fill_(-1.0)    # beta (bias) = -1
    bn.running_mean.fill_(0.0)   # 均值設為 0
    bn.running_var.fill_(1.0)    # **修正：標準差設為 1.0，確保 `sqrt(var + eps) = 1.0`**\

bn.eval()

# **輸入數據設為 2.0**
input_tensor = torch.full((1, C, H, W), 2.0)

# 執行 BatchNorm2d
output_tensor = bn(input_tensor)

# 轉換為 NumPy
output_np = output_tensor.detach().numpy().squeeze()

# 輸出結果
print("PyTorch BatchNorm2D Output:")
for c in range(C):
    print(f"Channel {c}:")
    print(output_np[c])

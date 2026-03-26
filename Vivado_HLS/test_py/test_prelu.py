import torch
import torch.nn as nn
import numpy as np

# 測試參數
C = 2   # 通道數
H = 4   # 高度
W = 4   # 寬度

# 建立 PReLU 層（設定固定參數）
prelu = nn.PReLU(C)

# 設定固定 α 參數
with torch.no_grad():
    prelu.weight[0] = 0.1  # 第一個通道 α = 0.1
    prelu.weight[1] = 0.5  # 第二個通道 α = 0.5

# 初始化輸入數據
input_tensor = torch.tensor([
    [[1.0, -1.0, 2.0, -2.0],
     [3.0, -3.0, 4.0, -4.0],
     [5.0, -5.0, 6.0, -6.0],
     [7.0, -7.0, 8.0, -8.0]],

    [[-1.0, 1.0, -2.0, 2.0],
     [-3.0, 3.0, -4.0, 4.0],
     [-5.0, 5.0, -6.0, 6.0],
     [-7.0, 7.0, -8.0, 8.0]]
]).unsqueeze(0)  # 增加 batch 維度

# 執行 PReLU
output_tensor = prelu(input_tensor)

# 轉換為 NumPy
output_np = output_tensor.detach().numpy().squeeze()

# 輸出結果
print("PyTorch PReLU Output:")
for c in range(C):
    print(f"Channel {c}:")
    print(output_np[c])

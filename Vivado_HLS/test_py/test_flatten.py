import torch
import numpy as np

# 測試參數
TEST_C = 2   # 測試通道數
TEST_H = 3   # 測試高度
TEST_W = 3   # 測試寬度

# 建立 PyTorch Flatten 層
flatten = torch.nn.Flatten()

# 初始化輸入特徵圖
input_tensor = torch.tensor([
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0],
     [7.0, 8.0, 9.0]],

    [[10.0, 11.0, 12.0],
     [13.0, 14.0, 15.0],
     [16.0, 17.0, 18.0]]
]).unsqueeze(0)  # 增加 batch 維度

# 執行 Flatten
output_tensor = flatten(input_tensor)

# 轉換為 NumPy
output_np = output_tensor.detach().numpy().squeeze()

# 輸出結果
print("PyTorch Flatten Output:")
print(output_np)

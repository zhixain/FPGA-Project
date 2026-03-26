import torch
import numpy as np

# 測試參數
TEST_IN_SIZE = 4   # 測試輸入大小
TEST_OUT_SIZE = 3  # 測試輸出大小

# 建立 PyTorch Linear 層 (關閉 Bias)
linear = torch.nn.Linear(TEST_IN_SIZE, TEST_OUT_SIZE, bias=False)

# 設定固定權重
with torch.no_grad():
    linear.weight.copy_(torch.tensor([
        [1.0, 0.5, 0.0, -0.5],
        [-0.5, 1.0, 0.5, 0.0],
        [0.0, -0.5, 1.0, 0.5]
    ]))

# 初始化輸入向量
input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])

# 執行 Linear
output_tensor = linear(input_tensor)

# 轉換為 NumPy
output_np = output_tensor.detach().numpy()

# 輸出結果
print("PyTorch Linear Output (No Bias):")
print(output_np)

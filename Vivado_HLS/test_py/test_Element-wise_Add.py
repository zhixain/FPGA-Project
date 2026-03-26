import numpy as np
import torch

# 測試參數
TEST_SIZE = 5  # 測試輸入大小

# 設定 Shortcut（跳躍連接輸入）
input1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# 設定卷積運算後的輸出
input2 = torch.tensor([0.5, -0.5, 1.5, -1.5, 0.0])

# 執行 Element-wise Add（跳躍連接）
output = input1 + input2

# 輸出結果
print("PyTorch Skip Connection Output:")
print(output.numpy())

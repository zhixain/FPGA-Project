import torch
import numpy as np

# 測試參數
TEST_SIZE = 5  # 測試輸入大小

# 建立 PyTorch BatchNorm1d 層
bn = torch.nn.BatchNorm1d(TEST_SIZE, affine=True)

# 設定固定 scale 和 bias
with torch.no_grad():
    bn.weight.copy_(torch.tensor([2.0, 0.5, -1.0, 1.5, 1.0]))  # scale
    bn.bias.copy_(torch.tensor([1.0, -1.0, 0.5, -0.5, 0.0]))  # bias
    bn.running_mean.fill_(0.0)   # 均值設為 0
    bn.running_var.fill_(1.0)    # 標準差設為 1 (使得 std + eps = 1)

# **強制進入推論模式 (inference mode)**
bn.eval()

# **修正輸入形狀**: (batch_size, num_features)
input_tensor = torch.tensor([[1.0, -1.0, 2.0, -2.0, 0.0]])  # shape: (1, TEST_SIZE)

# 執行 BatchNorm1d
output_tensor = bn(input_tensor)

# 轉換為 NumPy
output_np = output_tensor.detach().numpy().squeeze()  # 移除 batch 維度

# 輸出結果
print("PyTorch BatchNorm1D Output:")
print(output_np)

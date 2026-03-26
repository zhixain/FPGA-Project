import torch
import torch.nn as nn
import numpy as np

# 設定參數
IN_C = 4
OUT_C = 4
IN_H = 4
IN_W = 4
K = 3
STRIDE = 1
PADDING = 1
GROUPS = 2

# 建立輸入資料（與 C++ 一樣：i % 4）
feature_in_np = np.array([i % 4 for i in range(IN_C * IN_H * IN_W)], dtype=np.float32)
feature_in_np = feature_in_np.reshape(1, IN_C, IN_H, IN_W)  # NCHW 格式

# 建立權重資料（與 C++ 一樣：i * 0.1）
weight_np = np.array([i * 0.1 for i in range(OUT_C * (IN_C // GROUPS) * K * K)], dtype=np.float32)
weight_np = weight_np.reshape(OUT_C, IN_C // GROUPS, K, K)

# 建立 PyTorch conv2d layer
conv = nn.Conv2d(in_channels=IN_C, out_channels=OUT_C, kernel_size=K, stride=STRIDE, padding=PADDING, groups=GROUPS, bias=False)

# 把我們自定的權重套進去（覆蓋 PyTorch 的預設初始化）
with torch.no_grad():
    conv.weight.copy_(torch.from_numpy(weight_np))

# 執行卷積
feature_in_tensor = torch.from_numpy(feature_in_np)
feature_out_tensor = conv(feature_in_tensor)

# 輸出結果
print("Grouped Convolution Output:")
feature_out_np = feature_out_tensor.detach().numpy()
for c in range(OUT_C):
    print(f"Channel {c}:")
    for h in range(feature_out_np.shape[2]):
        for w in range(feature_out_np.shape[3]):
            print(f"{feature_out_np[0, c, h, w]:.4f}", end=" ")
        print()

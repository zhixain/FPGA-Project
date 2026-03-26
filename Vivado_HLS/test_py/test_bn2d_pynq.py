import torch
import torch.nn as nn

torch.manual_seed(0)

bn = nn.BatchNorm2d(num_features=3)
bn.eval()

with torch.no_grad():
    bn.running_mean.copy_(torch.tensor([0.5, 1.0, 1.5]))
    bn.running_var.copy_(torch.tensor([1.0, 1.0, 1.0]))
    bn.weight.copy_(torch.tensor([1.0, 1.0, 1.0]))  # γ
    bn.bias.copy_(torch.tensor([0.0, 0.0, 0.0]))    # β

input_tensor = torch.tensor(
    [[[[1.0, 2.0],
       [3.0, 4.0]],
      [[2.0, 4.0],
       [6.0, 8.0]],
      [[10.0, 20.0],
       [30.0, 40.0]]]],
    dtype=torch.float32
)

output = bn(input_tensor)

print("輸入資料:")
print(input_tensor)
print("\n輸出資料:")
print(output)

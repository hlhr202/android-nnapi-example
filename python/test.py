import torch

ten1_2x3 = torch.tensor([[1, 2, 3], [4, 5, 6]])
ten2_3x2 = torch.tensor([[1, 2], [3, 4], [5, 6]])

result = torch.matmul(ten1_2x3, ten2_3x2)

print(result)
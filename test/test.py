import torch
a = torch.triu(torch.full((1, 1), float("-inf")), diagonal=0)
print(a)
import torch

"""
t = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
t2 = torch.gather(t, 0, torch.tensor([[[0, 0], [0, 0]], [[0, 1], [0, 0]]]))
print(t2)
"""

input = torch.zeros((200, 1), dtype=torch.float32)
input.random_()
index = torch.zeros((200, 1), dtype=torch.long)
index.random_(to=2)

output = torch.gather(input, 1, index)

print(output)
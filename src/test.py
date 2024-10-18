import torch

total = torch.cuda.get_device_properties(0).total_memory
a = torch.cuda.memory_allocated(0)
print(total - a)

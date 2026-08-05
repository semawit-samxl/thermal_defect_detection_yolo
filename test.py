import torch
print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.device_count())

x=torch.tensor([1,0])
x=x.cuda
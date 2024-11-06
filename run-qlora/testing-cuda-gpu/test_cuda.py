import torch

# Is Cuda available?
print(f"Is Cuda available: {torch.cuda.is_available()}")

# How many are GPU's available?
num_gpus = torch.cuda.device_count()
print(f"Amount of GPU's available: {num_gpus}")

for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Current GPU device used
print(f"Current GPU used: {torch.cuda.current_device()}")
print(f"Its device: {torch.cuda.device(0)}")
print(f"Device name used GPU0: {torch.cuda.get_device_name(0)}")

# Defining the GPU that you want to use
device = torch.device("cuda:1")
print(f"GPU choice: {device}")

# Current GPU device used
print(f"Current GPU used: {torch.cuda.current_device()}")
print(f"Its device: {torch.cuda.device(1)}")
print(f"Device name used GPU1: {torch.cuda.get_device_name(1)}")

import torch

# Is Cuda available?
print(f"Is Cuda available: {torch.cuda.is_available()}")
print()

# Checking whether GPU or CPU is used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"GPU choice: {device}")
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
print()

# How many are GPU's available?
num_gpus = torch.cuda.device_count()
print(f"Amount of GPU's available: {num_gpus}")
print()

for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
print()

# Current GPU device used
print(f"Current GPU used: {torch.cuda.current_device()}")
print(f"Its device: {torch.cuda.device(0)}")
print(f"Device name used GPU0: {torch.cuda.get_device_name(0)}")
print()

# Defining the GPU that you want to use
device = torch.device("cuda:1")
print(f"GPU choice: {device}")
print()

# Current GPU device used
print(f"Current GPU used: {torch.cuda.current_device()}")
print(f"Its device: {torch.cuda.device(1)}")
print(f"Device name used GPU1: {torch.cuda.get_device_name(1)}")
print()

import torch


if torch.cuda.is_available():
    print("GPU (CUDA) が利用可能です。")
else:
    print("GPU (CUDA) は利用できません。")

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")
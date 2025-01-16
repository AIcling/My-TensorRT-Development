# env_info.py
import torch
import torch_tensorrt
import tensorrt as trt

print("=== Software Versions ===")
print(f"Python version       : {torch.sys.version.split()[0]}")
print(f"Torch version        : {torch.__version__}")
print(f"CUDA version (PyTorch): {torch.version.cuda}")
print(f"Torch-TensorRT version: {torch_tensorrt.__version__}")
print(f"TensorRT version      : {trt.__version__}")

print("\n=== GPU Info ===")
has_cuda = torch.cuda.is_available()
print(f"Is CUDA available?   : {has_cuda}")
if has_cuda:
    gpu_idx = 0 
    print(f"Current GPU name     : {torch.cuda.get_device_name(gpu_idx)}")
    print(f"Device capability    : {torch.cuda.get_device_capability(gpu_idx)}")
    print(f"Device count         : {torch.cuda.device_count()}")
else:
    print("No CUDA-enabled GPU found.")

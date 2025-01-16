# infer.py
import time
import torch
import torch.nn.functional as F
import torch_tensorrt.logging as trt_logging

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj


device = torch.device("cuda")  # 若编译时是 CPU，可改为 'cpu'

# 1. 加载数据
dataset = Planetoid(root='data/Planetoid', name='Cora')
data = dataset[0]

x_input = data.x.to(device, torch.float32)         
adj_input = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze(0).to(device, torch.float32)
print("[infer.py] x_input.shape =", x_input.shape)
print("[infer.py] adj_input.shape =", adj_input.shape)

# 2. 加载编译好的 Torch-TensorRT 模型
trt_ts_module = torch.jit.load("gcn_cora_trt.ts").eval().to(device)
print("[infer.py] Loaded gcn_cora_trt.ts")

# 3. 推理并计时
with torch.no_grad():
    # 让 CUDA 上的 kernel 同步，以确保计时准确
    torch.cuda.synchronize(device)
    start_time = time.time()
    
    # 执行推理
    output = trt_ts_module(x_input, adj_input)
    
    torch.cuda.synchronize(device)
    end_time = time.time()
    
    # 推理耗时 (秒)
    inference_time = end_time - start_time
    print(f"[infer.py] Inference time = {inference_time * 1000:.3f} ms")

    # 4. 计算准确率
    preds = output.argmax(dim=1)
    data.y = data.y.to(device)
    test_mask = data.test_mask
    correct = preds[test_mask].eq(data.y[test_mask]).sum().item()
    test_acc = correct / test_mask.sum().item()
    print(f"[infer.py] Torch-TensorRT Inference Test Accuracy: {test_acc:.4f}")

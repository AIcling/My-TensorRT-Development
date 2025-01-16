# export.py
import torch
import torch.nn.functional as F
import torch_tensorrt
import torch_tensorrt.logging as trt_logging

from torch.nn import Linear
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj

# trt_logging.set_reportable_log_level(trt_logging.Level.Debug)

class SimplifiedGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.linear1 = Linear(in_channels, hidden_channels)
        self.linear2 = Linear(hidden_channels, out_channels)
    def forward(self, x, adj):
        x = torch.matmul(adj, x)   # (N,N)@(N, in_channels)
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.matmul(adj, x)   # (N,N)@(N, hidden_channels)
        x = self.linear2(x)
        return x

# 加载数据
dataset = Planetoid(root='data/Planetoid', name='Cora')
data = dataset[0]
N = data.x.size(0)      # 2708
in_channels = data.x.size(1)   # 1433
out_channels = dataset.num_classes   # 7
adj_dense = to_dense_adj(data.edge_index, max_num_nodes=N).squeeze(0)
data.adj = adj_dense

print("[export.py] data.x.shape =", data.x.shape)
print("[export.py] data.adj.shape =", data.adj.shape)

# 初始化模型，加载权重
model = SimplifiedGCN(in_channels, 16, out_channels)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

device = torch.device("cuda")   
model.to(device)

x_example = data.x.to(device, torch.float32)      # [2708, 1433]
adj_example = data.adj.to(device, torch.float32)  # [2708, 2708]

print("[export.py] x_example:", x_example.shape)
print("[export.py] adj_example:", adj_example.shape)

model.training = False  # 关闭 dropout
traced_module = torch.jit.trace(
    model,
    (x_example, adj_example),
    check_trace=False
)
traced_module.eval()

# Torch-TensorRT compile (fixed shape)
print("[export.py] Start Torch-TensorRT compile with fixed shape ...")

trt_ts_module = torch_tensorrt.compile(
    traced_module,
    ir="torchscript",
    enabled_precisions={torch.float},
    truncate_long_and_double=True,
    debug=True,
    inputs=[
        # x: [2708, 1433]
        torch_tensorrt.Input(
            min_shape=(N, in_channels),
            opt_shape=(N, in_channels),
            max_shape=(N, in_channels),
            dtype=torch.float
        ),
        # adj: [2708, 2708]
        torch_tensorrt.Input(
            min_shape=(N, N),
            opt_shape=(N, N),
            max_shape=(N, N),
            dtype=torch.float
        ),
    ]
)

torch.jit.save(trt_ts_module, "gcn_cora_trt.ts")
print("[export.py] Torch-TensorRT compile done & saved to gcn_cora_trt.ts")

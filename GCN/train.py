# train.py
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj

hidden_channels = 16
learning_rate = 0.01
weight_decay = 5e-4
epochs = 200
dataset_name = "Cora"

dataset = Planetoid(root='data/Planetoid', name=dataset_name)
data = dataset[0]

# ---------- 1. 把 edge_index 转成稠密邻接矩阵 ----------
# 这样后续就不再调用 to_dense_adj
adj_dense = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze(0)
data.adj = adj_dense  # 存为 data 的一个属性

print(f"[train.py] data.x.shape       = {data.x.shape}")       # [2708, 1433]
print(f"[train.py] data.edge_index.shape = {data.edge_index.shape}") # [2, 10556]
print(f"[train.py] data.adj.shape     = {data.adj.shape}")     # [2708, 2708]

class SimplifiedGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.linear1 = Linear(in_channels, hidden_channels)
        self.linear2 = Linear(hidden_channels, out_channels)
    def forward(self, x, adj):
        # 不再调用 to_dense_adj！
        x = torch.matmul(adj, x)        # (N, N) @ (N, in_channels)
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.matmul(adj, x)        # (N, N) @ (N, hidden_channels)
        x = self.linear2(x)
        return x

model = SimplifiedGCN(
    in_channels = data.x.size(1),   # 1433
    hidden_channels = hidden_channels,
    out_channels = dataset.num_classes  # 7
)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def train_one_epoch():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj)  # 直接传 x 和 adj
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    out = model(data.x, data.adj)
    preds = out.argmax(dim=1)
    accs = []
    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        mask = getattr(data, mask_name)
        correct = preds[mask].eq(data.y[mask]).sum().item()
        acc = correct / mask.sum().item()
        accs.append(acc)
    return accs

best_val_acc = 0
for epoch in range(1, epochs + 1):
    loss = train_one_epoch()
    train_acc, val_acc, test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.4f}, "
              f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

model.load_state_dict(torch.load("best_model.pt"))
train_acc, val_acc, test_acc = test()
print(f"[train.py] Best model -> train: {train_acc:.4f}, val: {val_acc:.4f}, test: {test_acc:.4f}")

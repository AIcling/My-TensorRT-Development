# Path to High-Performance Inference: TensorRT Projects

### Building...

## Environment

- **Python**: 3.10.15
- **PyTorch**: 2.5.1+cu124
- **Torch-TensorRT**: 2.5.0
- **TensorRT**: 10.3.0
- **CUDA (PyTorch)**: 12.4
- **GPU Model**: NVIDIA GeForce RTX 3080 Ti Laptop GPU
- **OS**: Ubuntu 22.04

## [Vanilla GCN (Cora) + Torch-TensorRT](./GCN)

This repository demonstrates a **vanilla GCN** on the **Cora** dataset, converted into a dense adjacency version and deployed with **Torch-TensorRT** for acceleration.

### Key Steps

1. **Training (train.py)**

   - Uses a simplified GCN model with dense adjacency matrix, **avoiding PyG's scatter operations**.
   - Saves the best model weights in `best_model.pt`.
2. **Export (export.py)**

   - Loads `best_model.pt` and transforms it into a Torch-TensorRT–compiled model, `gcn_cora_trt.ts`.
   - Uses **fixed shapes** for `(x, adj)` to avoid shape mismatch issues.
3. **Inference (infer.py)**

   - Loads `gcn_cora_trt.ts`, feeds in `(x, adj)` of the same shapes used during export.
   - Prints the **inference time** and **test accuracy**.

### Run

```bash
# 1. Train
python train.py
# 2. Export + Compile
python export.py
# 3. Infer
python infer.py
```

### Results

```bash
[infer.py] Inference time = 148.722 ms
[infer.py] Torch-TensorRT Inference Test Accuracy: 0.7660
```

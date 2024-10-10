import time
import torch
import numpy as np
from model import load_model
from infer import load_engine, infer
from torchvision import datasets, transforms

def test_pytorch(model, device, test_loader):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
    end_time = time.time()
    avg_time = (end_time - start_time) / len(test_loader)
    print(f"PyTorch average infer time:{avg_time * 1000:.4f} ms")

def test_tensorrt(engine, test_loader):
    start_time = time.time()
    for data, _ in test_loader:
        image = data.numpy()
        output = infer(engine, image)
    end_time = time.time()
    avg_time = (end_time - start_time) / len(test_loader)
    print(f"TensorRT average infer time:{avg_time * 1000:.4f} ms")

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('mnist_cnn.pth', device)
    test_pytorch(model, device, test_loader)

    engine = load_engine('mnist_cnn.engine')
    test_tensorrt(engine, test_loader)

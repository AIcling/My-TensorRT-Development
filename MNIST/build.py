import torch
import tensorrt as trt
from model import load_model
import pycuda.driver as cuda
import pycuda.autoinit

def export_to_onnx(model_path, onnx_file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    dummy_input = torch.randn(1, 1, 28, 28, device=device)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

def build_engine(onnx_file_path, engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(EXPLICIT_BATCH) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        builder.max_workspace_size = 1 << 30  
        builder.fp16_mode = True  

        engine = builder.build_cuda_engine(network)
        if engine is not None:
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
        return engine

if __name__ == "__main__":
    model_path = 'mnist_cnn.pth'
    onnx_file_path = 'mnist_cnn.onnx'
    engine_file_path = 'mnist_cnn.engine'

    export_to_onnx(model_path, onnx_file_path)
    build_engine(onnx_file_path, engine_file_path)

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
from torchvision import transforms
from PIL import Image

def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        print(f"TensorRT engine are ready in:{engine_file_path}")
        return engine

def infer(engine, image):
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = [], [], [], cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})

    np.copyto(inputs[0]['host'], image.ravel())

    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()

    output_data = outputs[0]['host']
    return output_data

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(image_path)
    image = transform(image).numpy()
    image = np.expand_dims(image, axis=0)  
    return image

if __name__ == "__main__":
    engine_file_path = 'mnist_cnn.engine'
    image_path = 'data/test_image.png'  

    engine = load_engine(engine_file_path)
    image = preprocess_image(image_path)
    output = infer(engine, image)
    output = torch.from_numpy(output).reshape(1, -1)
    pred = torch.argmax(output, dim=1)
    print(f"res:{pred.item()}")

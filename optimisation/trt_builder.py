import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, precision='fp16', batch_size=512, tile_size=256):
    """
    Builds a TensorRT engine from an ONNX file.
    precision: 'fp32', 'fp16', 'int8', or 'fp8'
    """
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    config = builder.create_builder_config()
    # Let TensorRT use up to 4GB of workspace for layer tactics
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
    
    if precision == 'fp16':
        if not builder.platform_has_fast_fp16:
            print("Warning: Platform does not have fast FP16. Using FP32.")
        else:
            config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        if not builder.platform_has_fast_int8:
            print("Warning: Platform does not have fast INT8. Using FP32.")
        else:
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_flag(trt.BuilderFlag.FP16)
            print("Note: INT8 typically requires a calibrator to be supplied in the config.")
    elif precision == 'fp8':
        try:
            config.set_flag(trt.BuilderFlag.FP8)
            print("Enabled FP8 natively.")
        except AttributeError:
            print("Warning: FP8 flag not found in this TRT version. Fallback to FP16.")
            config.set_flag(trt.BuilderFlag.FP16)
            
    # Parse ONNX
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
            
    # Define optimization profile for dynamic batch sizes
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, tile_size, tile_size), (batch_size//2, 3, tile_size, tile_size), (batch_size, 3, tile_size, tile_size))
    config.add_optimization_profile(profile)
    
    print(f"Building {precision.upper()} engine. This may take a few minutes...")
    engine_bytes = builder.build_serialized_network(network, config)
    
    if engine_bytes is None:
        print("Failed to build engine.")
        return None
        
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
        
    print(f"Successfully built and saved engine to {engine_path}")
    return True

import tensorrt as trt
import torch
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTWrapper:
    def __init__(self, engine_path):
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream()
        
        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            # Using max profile shape to allocate memory
            shape = self.engine.get_tensor_profile_shape(name, 0)[2] if is_input else self.engine.get_tensor_shape(name)
            # If output shape has dynamic batch dimension
            if shape[0] == -1: 
                shape = (self.engine.get_tensor_profile_shape(self.engine.get_tensor_name(0), 0)[2][0],) + shape[1:]
                
            # Create a torch tensor of the appropriate type and shape
            if dtype == np.float32:
                torch_dtype = torch.float32
            elif dtype == np.float16:
                torch_dtype = torch.float16
            elif dtype == np.int32:
                torch_dtype = torch.int32
            elif dtype == np.int8:
                torch_dtype = torch.int8
            else:
                torch_dtype = torch.float32
                
            allocation = torch.empty(tuple(shape), dtype=torch_dtype, device='cuda')
            self.allocations.append(allocation)
            
            binding = {
                'index': i,
                'name': name,
                'dtype': torch_dtype,
                'shape': shape,
                'allocation': allocation
            }
            
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
                
        self.graph = None

    def capture_graph(self, dummy_input: torch.Tensor):
        self.context.set_input_shape(self.inputs[0]['name'], tuple(dummy_input.shape))
        self.inputs[0]['allocation'].copy_(dummy_input)
        
        for alloc, name in zip(self.allocations, [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]):
            self.context.set_tensor_address(name, alloc.data_ptr())
            
        # Warmup
        self.context.execute_async_v3(stream_ptr=self.stream.cuda_stream)
        self.stream.synchronize()
        
        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.context.execute_async_v3(stream_ptr=self.stream.cuda_stream)

    def infer_graph(self, input_tensor: torch.Tensor) -> list:
        # Copy input_tensor to the input allocation buffer
        self.inputs[0]['allocation'].copy_(input_tensor)
        
        # Replay the captured CUDA graph
        self.graph.replay()
        
        # Return clones of the outputs to prevent overwriting
        return [out['allocation'].clone() for out in self.outputs]

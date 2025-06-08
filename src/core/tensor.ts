import { Tensor } from './types';
import { WebGPUContext } from './webgpu';

export class TensorOps {
  private static gpu = WebGPUContext.getInstance();
  
  static async create(
    data: Float32Array | number[],
    shape: number[],
    device: 'cpu' | 'gpu' = 'gpu'
  ): Promise<Tensor> {
    const flatData = data instanceof Float32Array ? data : new Float32Array(data);
    const strides = this.computeStrides(shape);
    
    let gpuBuffer: GPUBuffer | undefined;
    
    if (device === 'gpu') {
      await this.gpu.initialize();
      const bufferSize = flatData.byteLength;
      gpuBuffer = await this.gpu.createBuffer(
        bufferSize,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
      );
      await this.gpu.writeBuffer(gpuBuffer, flatData);
    }
    
    return {
      data: flatData,
      shape,
      strides,
      device,
      gpuBuffer
    };
  }
  
  static zeros(shape: number[], device: 'cpu' | 'gpu' = 'gpu'): Promise<Tensor> {
    const size = shape.reduce((a, b) => a * b, 1);
    return this.create(new Float32Array(size), shape, device);
  }
  
  static ones(shape: number[], device: 'cpu' | 'gpu' = 'gpu'): Promise<Tensor> {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size).fill(1);
    return this.create(data, shape, device);
  }
  
  static random(shape: number[], device: 'cpu' | 'gpu' = 'gpu'): Promise<Tensor> {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    
    for (let i = 0; i < size; i++) {
      data[i] = (Math.random() - 0.5) * 2 * Math.sqrt(6 / (shape[0] + shape[1]));
    }
    
    return this.create(data, shape, device);
  }
  
  static async toGPU(tensor: Tensor): Promise<Tensor> {
    if (tensor.device === 'gpu') {
      return tensor;
    }
    
    try {
      await this.gpu.initialize();
      const bufferSize = tensor.data.byteLength;
      const gpuBuffer = await this.gpu.createBuffer(
        bufferSize,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
      );
      await this.gpu.writeBuffer(gpuBuffer, tensor.data);
      
      return {
        ...tensor,
        device: 'gpu',
        gpuBuffer
      };
    } catch (error) {
      console.warn('WebGPU not available, keeping tensor on CPU:', error);
      return tensor;
    }
  }
  
  static async toCPU(tensor: Tensor): Promise<Tensor> {
    if (tensor.device === 'cpu') {
      return tensor;
    }
    
    if (!tensor.gpuBuffer) {
      throw new Error('GPU tensor missing buffer');
    }
    
    const data = await this.gpu.readBuffer(tensor.gpuBuffer, tensor.data.byteLength);
    
    return {
      ...tensor,
      data,
      device: 'cpu',
      gpuBuffer: undefined
    };
  }
  
  static reshape(tensor: Tensor, newShape: number[]): Tensor {
    const oldSize = tensor.shape.reduce((a, b) => a * b, 1);
    const newSize = newShape.reduce((a, b) => a * b, 1);
    
    if (oldSize !== newSize) {
      throw new Error(`Cannot reshape tensor of size ${oldSize} to size ${newSize}`);
    }
    
    return {
      ...tensor,
      shape: newShape,
      strides: this.computeStrides(newShape)
    };
  }
  
  static slice(tensor: Tensor, start: number[], end: number[]): Tensor {
    if (tensor.device === 'gpu') {
      throw new Error('GPU slicing requires CPU conversion first - use TensorOps.toCPU(tensor) then slice');
    }
    
    const newShape = start.map((s, i) => end[i] - s);
    const newSize = newShape.reduce((a, b) => a * b, 1);
    const newData = new Float32Array(newSize);
    
    const srcIdx = 0;
    const dstIdx = 0;
    
    const copySlice = (dim: number, srcOffset: number, dstOffset: number) => {
      if (dim === tensor.shape.length) {
        newData[dstOffset] = tensor.data[srcOffset];
        return;
      }
      
      for (let i = start[dim]; i < end[dim]; i++) {
        copySlice(
          dim + 1,
          srcOffset + i * tensor.strides[dim],
          dstOffset + (i - start[dim]) * this.computeStrides(newShape)[dim]
        );
      }
    };
    
    copySlice(0, 0, 0);
    
    return {
      data: newData,
      shape: newShape,
      strides: this.computeStrides(newShape),
      device: 'cpu'
    };
  }
  
  private static computeStrides(shape: number[]): number[] {
    const strides = new Array(shape.length);
    let stride = 1;
    
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i];
    }
    
    return strides;
  }
  
  static async add(a: Tensor, b: Tensor): Promise<Tensor> {
    if (a.device !== b.device) {
      throw new Error('Tensors must be on same device');
    }
    
    if (a.device === 'cpu') {
      const result = new Float32Array(a.data.length);
      for (let i = 0; i < a.data.length; i++) {
        result[i] = a.data[i] + b.data[i];
      }
      return this.create(result, a.shape, 'cpu');
    }
    
    const { ElementwiseKernel } = await import('../kernels/elementwise');
    return ElementwiseKernel.add(a, b);
  }
  
  static async multiply(a: Tensor, b: Tensor): Promise<Tensor> {
    if (a.device !== b.device) {
      throw new Error('Tensors must be on same device');
    }
    
    if (a.device === 'cpu') {
      const result = new Float32Array(a.data.length);
      for (let i = 0; i < a.data.length; i++) {
        result[i] = a.data[i] * b.data[i];
      }
      return this.create(result, a.shape, 'cpu');
    }
    
    const { ElementwiseKernel } = await import('../kernels/elementwise');
    return ElementwiseKernel.multiply(a, b);
  }
}
import { Tensor } from '../core/types';
import { WebGPUContext } from '../core/webgpu';
import { TensorOps } from '../core/tensor';

const elementwiseShader = `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> size: u32;

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= size) { return; }
  
  output[idx] = a[idx] + b[idx];
}

@compute @workgroup_size(256)
fn multiply(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= size) { return; }
  
  output[idx] = a[idx] * b[idx];
}

@compute @workgroup_size(256)
fn subtract(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= size) { return; }
  
  output[idx] = a[idx] - b[idx];
}

@compute @workgroup_size(256)
fn divide(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= size) { return; }
  
  output[idx] = a[idx] / b[idx];
}

@compute @workgroup_size(256)
fn scalar_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= size) { return; }
  
  output[idx] = a[idx] + b[0];
}

@compute @workgroup_size(256)
fn bias_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= size) { return; }
  
  let batch_size = size / arrayLength(&b);
  let feature_idx = idx % arrayLength(&b);
  
  output[idx] = a[idx] + b[feature_idx];
}

@compute @workgroup_size(256)
fn sum_reduction(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let batch_size = size / arrayLength(&output);
  
  if (idx >= arrayLength(&output)) { return; }
  
  var sum = 0.0;
  for (var i = 0u; i < batch_size; i = i + 1u) {
    sum = sum + a[i * arrayLength(&output) + idx];
  }
  
  output[idx] = sum;
}
`;

export class ElementwiseKernel {
  private static gpu = WebGPUContext.getInstance();
  private static pipelineCache = new Map<string, GPUComputePipeline>();
  
  static async add(a: Tensor, b: Tensor): Promise<Tensor> {
    if (a.device === 'cpu' || b.device === 'cpu') {
      return this.cpuAdd(a, b);
    }
    
    return this.gpuElementwise(a, b, 'add');
  }
  
  static async multiply(a: Tensor, b: Tensor): Promise<Tensor> {
    if (a.device === 'cpu' || b.device === 'cpu') {
      return this.cpuMultiply(a, b);
    }
    
    return this.gpuElementwise(a, b, 'multiply');
  }
  
  static async subtract(a: Tensor, b: Tensor): Promise<Tensor> {
    if (a.device === 'cpu' || b.device === 'cpu') {
      return this.cpuSubtract(a, b);
    }
    
    return this.gpuElementwise(a, b, 'subtract');
  }
  
  static async addBias(input: Tensor, bias: Tensor): Promise<Tensor> {
    if (input.device === 'cpu' || bias.device === 'cpu') {
      return this.cpuAddBias(input, bias);
    }
    
    return this.gpuBiasAdd(input, bias);
  }
  
  static async sumGradients(gradients: Tensor, outputShape: number[]): Promise<Tensor> {
    if (gradients.device === 'cpu') {
      return this.cpuSumGradients(gradients, outputShape);
    }
    
    return this.gpuSumReduction(gradients, outputShape);
  }
  
  private static async cpuAdd(a: Tensor, b: Tensor): Promise<Tensor> {
    const result = new Float32Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      result[i] = a.data[i] + b.data[i];
    }
    return TensorOps.create(result, a.shape, 'cpu');
  }
  
  private static async cpuMultiply(a: Tensor, b: Tensor): Promise<Tensor> {
    const result = new Float32Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      result[i] = a.data[i] * b.data[i];
    }
    return TensorOps.create(result, a.shape, 'cpu');
  }
  
  private static async cpuSubtract(a: Tensor, b: Tensor): Promise<Tensor> {
    const result = new Float32Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      result[i] = a.data[i] - b.data[i];
    }
    return TensorOps.create(result, a.shape, 'cpu');
  }
  
  private static async cpuAddBias(input: Tensor, bias: Tensor): Promise<Tensor> {
    const result = new Float32Array(input.data.length);
    const batchSize = input.shape[0];
    const outputSize = bias.shape[0];
    
    for (let b = 0; b < batchSize; b++) {
      for (let o = 0; o < outputSize; o++) {
        result[b * outputSize + o] = input.data[b * outputSize + o] + bias.data[o];
      }
    }
    
    return TensorOps.create(result, input.shape, 'cpu');
  }
  
  private static async cpuSumGradients(gradients: Tensor, outputShape: number[]): Promise<Tensor> {
    const batchSize = gradients.shape[0];
    const outputSize = outputShape[0];
    const result = new Float32Array(outputSize).fill(0);
    
    for (let b = 0; b < batchSize; b++) {
      for (let o = 0; o < outputSize; o++) {
        result[o] += gradients.data[b * outputSize + o];
      }
    }
    
    return TensorOps.create(result, outputShape, 'cpu');
  }
  
  private static async gpuElementwise(a: Tensor, b: Tensor, operation: string): Promise<Tensor> {
    const device = this.gpu.getDevice();
    const size = a.data.length;
    
    const cacheKey = `elementwise_${operation}_${size}`;
    let pipeline = this.pipelineCache.get(cacheKey);
    
    if (!pipeline) {
      const bindGroupLayout = this.gpu.createBindGroupLayout([
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' as GPUBufferBindingType }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' as GPUBufferBindingType }
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' as GPUBufferBindingType }
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' as GPUBufferBindingType }
        }
      ]);
      
      pipeline = this.gpu.createComputePipeline(
        elementwiseShader,
        bindGroupLayout,
        operation
      );
      
      this.pipelineCache.set(cacheKey, pipeline);
    }
    
    const sizeBuffer = await this.gpu.createBuffer(
      4,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    const sizeArray = new Uint32Array([size]);
    await this.gpu.writeBuffer(sizeBuffer, new Float32Array(sizeArray.buffer));
    
    const outputBuffer = await this.gpu.createBuffer(
      size * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: a.gpuBuffer! } },
        { binding: 1, resource: { buffer: b.gpuBuffer! } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: sizeBuffer } }
      ]
    });
    
    const workgroupSize = 256;
    const numWorkgroups = Math.ceil(size / workgroupSize);
    
    await this.gpu.dispatch(pipeline, bindGroup, [numWorkgroups, 1, 1]);
    
    return {
      data: new Float32Array(size),
      shape: a.shape,
      strides: a.strides,
      device: 'gpu',
      gpuBuffer: outputBuffer
    };
  }
  
  private static async gpuBiasAdd(input: Tensor, bias: Tensor): Promise<Tensor> {
    const device = this.gpu.getDevice();
    const size = input.data.length;
    
    const cacheKey = `bias_add_${size}_${bias.data.length}`;
    let pipeline = this.pipelineCache.get(cacheKey);
    
    if (!pipeline) {
      const bindGroupLayout = this.gpu.createBindGroupLayout([
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' as GPUBufferBindingType }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' as GPUBufferBindingType }
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' as GPUBufferBindingType }
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' as GPUBufferBindingType }
        }
      ]);
      
      pipeline = this.gpu.createComputePipeline(
        elementwiseShader,
        bindGroupLayout,
        'bias_add'
      );
      
      this.pipelineCache.set(cacheKey, pipeline);
    }
    
    const sizeBuffer = await this.gpu.createBuffer(
      4,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    const sizeArray = new Uint32Array([size]);
    await this.gpu.writeBuffer(sizeBuffer, new Float32Array(sizeArray.buffer));
    
    const outputBuffer = await this.gpu.createBuffer(
      size * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.gpuBuffer! } },
        { binding: 1, resource: { buffer: bias.gpuBuffer! } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: sizeBuffer } }
      ]
    });
    
    const workgroupSize = 256;
    const numWorkgroups = Math.ceil(size / workgroupSize);
    
    await this.gpu.dispatch(pipeline, bindGroup, [numWorkgroups, 1, 1]);
    
    return {
      data: new Float32Array(size),
      shape: input.shape,
      strides: input.strides,
      device: 'gpu',
      gpuBuffer: outputBuffer
    };
  }
  
  private static async gpuSumReduction(gradients: Tensor, outputShape: number[]): Promise<Tensor> {
    const device = this.gpu.getDevice();
    const outputSize = outputShape.reduce((a, b) => a * b, 1);
    
    const cacheKey = `sum_reduction_${gradients.data.length}_${outputSize}`;
    let pipeline = this.pipelineCache.get(cacheKey);
    
    if (!pipeline) {
      const bindGroupLayout = this.gpu.createBindGroupLayout([
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' as GPUBufferBindingType }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' as GPUBufferBindingType }
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' as GPUBufferBindingType }
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' as GPUBufferBindingType }
        }
      ]);
      
      pipeline = this.gpu.createComputePipeline(
        elementwiseShader,
        bindGroupLayout,
        'sum_reduction'
      );
      
      this.pipelineCache.set(cacheKey, pipeline);
    }
    
    const sizeBuffer = await this.gpu.createBuffer(
      4,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    const sizeArray = new Uint32Array([gradients.data.length]);
    await this.gpu.writeBuffer(sizeBuffer, new Float32Array(sizeArray.buffer));
    
    const outputBuffer = await this.gpu.createBuffer(
      outputSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    
    const dummyBuffer = await this.gpu.createBuffer(
      4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gradients.gpuBuffer! } },
        { binding: 1, resource: { buffer: dummyBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: sizeBuffer } }
      ]
    });
    
    const workgroupSize = 256;
    const numWorkgroups = Math.ceil(outputSize / workgroupSize);
    
    await this.gpu.dispatch(pipeline, bindGroup, [numWorkgroups, 1, 1]);
    
    return {
      data: new Float32Array(outputSize),
      shape: outputShape,
      strides: this.computeStrides(outputShape),
      device: 'gpu',
      gpuBuffer: outputBuffer
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
}
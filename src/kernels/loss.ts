import { Tensor } from '../core/types';
import { WebGPUContext } from '../core/webgpu';
import { TensorOps } from '../core/tensor';

const lossShader = `
@group(0) @binding(0) var<storage, read> predictions: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec2<u32>; // batch_size, num_classes

@compute @workgroup_size(256)
fn cross_entropy_loss(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let batch_idx = global_id.x;
  let batch_size = params.x;
  let num_classes = params.y;
  
  if (batch_idx >= batch_size) { return; }
  
  var loss = 0.0;
  for (var c = 0u; c < num_classes; c = c + 1u) {
    let pred_idx = batch_idx * num_classes + c;
    let target_idx = batch_idx * num_classes + c;
    
    let pred = max(predictions[pred_idx], 1e-15);
    loss = loss - targets[target_idx] * log(pred);
  }
  
  output[batch_idx] = loss;
}

@compute @workgroup_size(256)
fn cross_entropy_gradient(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let total_size = params.x * params.y;
  
  if (idx >= total_size) { return; }
  
  output[idx] = predictions[idx] - targets[idx];
}

@compute @workgroup_size(256)
fn tensor_sum(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx > 0u) { return; }
  
  var sum = 0.0;
  for (var i = 0u; i < arrayLength(&predictions); i = i + 1u) {
    sum = sum + predictions[i];
  }
  
  output[0] = sum;
}
`;

export class LossKernel {
  private static gpu = WebGPUContext.getInstance();
  private static pipelineCache = new Map<string, GPUComputePipeline>();
  
  static async crossEntropyLoss(predictions: Tensor, targets: Tensor): Promise<Tensor> {
    if (predictions.device === 'cpu') {
      return this.cpuCrossEntropyLoss(predictions, targets);
    }
    
    return this.gpuCrossEntropyLoss(predictions, targets);
  }
  
  static async crossEntropyGradient(predictions: Tensor, targets: Tensor): Promise<Tensor> {
    if (predictions.device === 'cpu') {
      return this.cpuCrossEntropyGradient(predictions, targets);
    }
    
    return this.gpuCrossEntropyGradient(predictions, targets);
  }
  
  static async tensorSum(tensor: Tensor): Promise<number> {
    if (tensor.device === 'cpu') {
      return tensor.data.reduce((sum, val) => sum + val, 0);
    }
    
    return this.gpuTensorSum(tensor);
  }
  
  private static async cpuCrossEntropyLoss(predictions: Tensor, targets: Tensor): Promise<Tensor> {
    const batchSize = predictions.shape[0];
    const numClasses = predictions.shape[1];
    const losses = new Float32Array(batchSize);
    
    for (let b = 0; b < batchSize; b++) {
      let loss = 0;
      for (let c = 0; c < numClasses; c++) {
        const predIdx = b * numClasses + c;
        const targetIdx = b * numClasses + c;
        
        const pred = Math.max(predictions.data[predIdx], 1e-15);
        loss -= targets.data[targetIdx] * Math.log(pred);
      }
      losses[b] = loss;
    }
    
    return TensorOps.create(losses, [batchSize], 'cpu');
  }
  
  private static async cpuCrossEntropyGradient(predictions: Tensor, targets: Tensor): Promise<Tensor> {
    const gradients = new Float32Array(predictions.data.length);
    
    for (let i = 0; i < predictions.data.length; i++) {
      gradients[i] = predictions.data[i] - targets.data[i];
    }
    
    return TensorOps.create(gradients, predictions.shape, 'cpu');
  }
  
  private static async gpuCrossEntropyLoss(predictions: Tensor, targets: Tensor): Promise<Tensor> {
    const device = this.gpu.getDevice();
    const batchSize = predictions.shape[0];
    const numClasses = predictions.shape[1];
    
    const cacheKey = `cross_entropy_loss_${batchSize}_${numClasses}`;
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
        lossShader,
        bindGroupLayout,
        'cross_entropy_loss'
      );
      
      this.pipelineCache.set(cacheKey, pipeline);
    }
    
    const paramsBuffer = await this.gpu.createBuffer(
      8,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    const paramsArray = new Uint32Array([batchSize, numClasses]);
    await this.gpu.writeBuffer(paramsBuffer, new Float32Array(paramsArray.buffer));
    
    const outputBuffer = await this.gpu.createBuffer(
      batchSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: predictions.gpuBuffer! } },
        { binding: 1, resource: { buffer: targets.gpuBuffer! } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } }
      ]
    });
    
    const workgroupSize = 256;
    const numWorkgroups = Math.ceil(batchSize / workgroupSize);
    
    await this.gpu.dispatch(pipeline, bindGroup, [numWorkgroups, 1, 1]);
    
    return {
      data: new Float32Array(batchSize),
      shape: [batchSize],
      strides: [1],
      device: 'gpu',
      gpuBuffer: outputBuffer
    };
  }
  
  private static async gpuCrossEntropyGradient(predictions: Tensor, targets: Tensor): Promise<Tensor> {
    const device = this.gpu.getDevice();
    const totalSize = predictions.data.length;
    const batchSize = predictions.shape[0];
    const numClasses = predictions.shape[1];
    
    const cacheKey = `cross_entropy_gradient_${batchSize}_${numClasses}`;
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
        lossShader,
        bindGroupLayout,
        'cross_entropy_gradient'
      );
      
      this.pipelineCache.set(cacheKey, pipeline);
    }
    
    const paramsBuffer = await this.gpu.createBuffer(
      8,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    const paramsArray = new Uint32Array([batchSize, numClasses]);
    await this.gpu.writeBuffer(paramsBuffer, new Float32Array(paramsArray.buffer));
    
    const outputBuffer = await this.gpu.createBuffer(
      totalSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: predictions.gpuBuffer! } },
        { binding: 1, resource: { buffer: targets.gpuBuffer! } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } }
      ]
    });
    
    const workgroupSize = 256;
    const numWorkgroups = Math.ceil(totalSize / workgroupSize);
    
    await this.gpu.dispatch(pipeline, bindGroup, [numWorkgroups, 1, 1]);
    
    return {
      data: new Float32Array(totalSize),
      shape: predictions.shape,
      strides: predictions.strides,
      device: 'gpu',
      gpuBuffer: outputBuffer
    };
  }
  
  private static async gpuTensorSum(tensor: Tensor): Promise<number> {
    const device = this.gpu.getDevice();
    
    const cacheKey = `tensor_sum_${tensor.data.length}`;
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
        lossShader,
        bindGroupLayout,
        'tensor_sum'
      );
      
      this.pipelineCache.set(cacheKey, pipeline);
    }
    
    const paramsBuffer = await this.gpu.createBuffer(
      8,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    await this.gpu.writeBuffer(paramsBuffer, new Float32Array([0, 0]));
    
    const outputBuffer = await this.gpu.createBuffer(
      4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    
    const dummyBuffer = await this.gpu.createBuffer(
      4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tensor.gpuBuffer! } },
        { binding: 1, resource: { buffer: dummyBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } }
      ]
    });
    
    await this.gpu.dispatch(pipeline, bindGroup, [1, 1, 1]);
    
    const result = await this.gpu.readBuffer(outputBuffer, 4);
    return result[0];
  }
}
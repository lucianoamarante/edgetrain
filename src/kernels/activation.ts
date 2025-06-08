import { Tensor } from '../core/types';
import { WebGPUContext } from '../core/webgpu';
import { TensorOps } from '../core/tensor';
const activationShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> size: u32;

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= size) { return; }
  
  output[idx] = max(input[idx], 0.0);
}

@compute @workgroup_size(256)
fn relu_backward(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @group(0) @binding(3) var<storage, read> grad_output: array<f32>
) {
  let idx = global_id.x;
  if (idx >= size) { return; }
  
  if (input[idx] > 0.0) {
    output[idx] = grad_output[idx];
  } else {
    output[idx] = 0.0;
  }
}

@compute @workgroup_size(256)
fn sigmoid(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= size) { return; }
  
  output[idx] = 1.0 / (1.0 + exp(-input[idx]));
}

@compute @workgroup_size(256)
fn sigmoid_backward(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @group(0) @binding(3) var<storage, read> grad_output: array<f32>
) {
  let idx = global_id.x;
  if (idx >= size) { return; }
  
  let s = output[idx];
  output[idx] = grad_output[idx] * s * (1.0 - s);
}

@compute @workgroup_size(256)
fn tanh_activation(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= size) { return; }
  
  let exp_2x = exp(2.0 * input[idx]);
  output[idx] = (exp_2x - 1.0) / (exp_2x + 1.0);
}

@compute @workgroup_size(256)
fn softmax(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let batch_idx = global_id.x;
  let batch_size = size / 10u;
  
  if (batch_idx >= batch_size) { return; }
  
  let offset = batch_idx * 10u;
  
  var max_val = input[offset];
  for (var i = 1u; i < 10u; i = i + 1u) {
    max_val = max(max_val, input[offset + i]);
  }
  
  var sum = 0.0;
  for (var i = 0u; i < 10u; i = i + 1u) {
    let exp_val = exp(input[offset + i] - max_val);
    output[offset + i] = exp_val;
    sum = sum + exp_val;
  }
  
  for (var i = 0u; i < 10u; i = i + 1u) {
    output[offset + i] = output[offset + i] / sum;
  }
}
`;

export type ActivationType = 'relu' | 'sigmoid' | 'tanh' | 'softmax';

export class ActivationKernel {
  private static gpu = WebGPUContext.getInstance();
  private static pipelineCache = new Map<string, GPUComputePipeline>();
  
  static async forward(input: Tensor, activation: ActivationType): Promise<Tensor> {
    if (input.device === 'cpu') {
      return this.cpuActivation(input, activation);
    }
    
    return this.gpuActivation(input, activation);
  }
  
  static async backward(
    gradOutput: Tensor,
    output: Tensor,
    activation: ActivationType
  ): Promise<Tensor> {
    if (gradOutput.device === 'cpu') {
      return this.cpuActivationBackward(gradOutput, output, activation);
    }
    
    return this.gpuActivationBackward(gradOutput, output, activation);
  }
  
  private static async cpuActivation(input: Tensor, activation: ActivationType): Promise<Tensor> {
    const size = input.data.length;
    const result = new Float32Array(size);
    
    switch (activation) {
    case 'relu':
      for (let i = 0; i < size; i++) {
        result[i] = Math.max(0, input.data[i]);
      }
      break;
        
    case 'sigmoid':
      for (let i = 0; i < size; i++) {
        result[i] = 1 / (1 + Math.exp(-input.data[i]));
      }
      break;
        
    case 'tanh':
      for (let i = 0; i < size; i++) {
        const exp2x = Math.exp(2 * input.data[i]);
        result[i] = (exp2x - 1) / (exp2x + 1);
      }
      break;
        
    case 'softmax':
      const batchSize = input.shape[0];
      const numClasses = input.shape[input.shape.length - 1];
        
      for (let b = 0; b < batchSize; b++) {
        const offset = b * numClasses;
        let maxVal = input.data[offset];
          
        for (let i = 1; i < numClasses; i++) {
          maxVal = Math.max(maxVal, input.data[offset + i]);
        }
          
        let sum = 0;
        for (let i = 0; i < numClasses; i++) {
          result[offset + i] = Math.exp(input.data[offset + i] - maxVal);
          sum += result[offset + i];
        }
          
        for (let i = 0; i < numClasses; i++) {
          result[offset + i] /= sum;
        }
      }
      break;
    }
    
    return TensorOps.create(result, input.shape, 'cpu');
  }
  
  private static async cpuActivationBackward(
    gradOutput: Tensor,
    output: Tensor,
    activation: ActivationType
  ): Promise<Tensor> {
    const size = gradOutput.data.length;
    const result = new Float32Array(size);
    
    switch (activation) {
    case 'relu':
      for (let i = 0; i < size; i++) {
        result[i] = output.data[i] > 0 ? gradOutput.data[i] : 0;
      }
      break;
        
    case 'sigmoid':
      for (let i = 0; i < size; i++) {
        const s = output.data[i];
        result[i] = gradOutput.data[i] * s * (1 - s);
      }
      break;
        
    case 'tanh':
      for (let i = 0; i < size; i++) {
        const t = output.data[i];
        result[i] = gradOutput.data[i] * (1 - t * t);
      }
      break;
        
    case 'softmax':
      // For softmax, we compute: grad_input = softmax_output * (grad_output - sum(grad_output * softmax_output))
      const batchSize = output.shape[0];
      const numClasses = output.shape[output.shape.length - 1];
        
      for (let b = 0; b < batchSize; b++) {
        const offset = b * numClasses;
          
        // Compute sum of grad_output * softmax_output
        let sum = 0;
        for (let i = 0; i < numClasses; i++) {
          sum += gradOutput.data[offset + i] * output.data[offset + i];
        }
          
        // Compute gradient
        for (let i = 0; i < numClasses; i++) {
          result[offset + i] = output.data[offset + i] * (gradOutput.data[offset + i] - sum);
        }
      }
      break;
    }
    
    return TensorOps.create(result, gradOutput.shape, 'cpu');
  }
  
  private static async gpuActivation(input: Tensor, activation: ActivationType): Promise<Tensor> {
    const device = this.gpu.getDevice();
    const size = input.data.length;
    
    const cacheKey = `activation_${activation}_${size}`;
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
          buffer: { type: 'storage' as GPUBufferBindingType }
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' as GPUBufferBindingType }
        }
      ]);
      
      pipeline = this.gpu.createComputePipeline(
        activationShader,
        bindGroupLayout,
        activation === 'tanh' ? 'tanh_activation' : activation
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
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: sizeBuffer } }
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
  
  private static async gpuActivationBackward(
    gradOutput: Tensor,
    output: Tensor,
    activation: ActivationType
  ): Promise<Tensor> {
    const device = this.gpu.getDevice();
    const size = gradOutput.data.length;
    
    const cacheKey = `activation_backward_${activation}_${size}`;
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
          buffer: { type: 'storage' as GPUBufferBindingType }
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' as GPUBufferBindingType }
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' as GPUBufferBindingType }
        }
      ]);
      
      pipeline = this.gpu.createComputePipeline(
        activationShader,
        bindGroupLayout,
        `${activation}_backward`
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
        { binding: 0, resource: { buffer: output.gpuBuffer! } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: sizeBuffer } },
        { binding: 3, resource: { buffer: gradOutput.gpuBuffer! } }
      ]
    });
    
    const workgroupSize = 256;
    const numWorkgroups = Math.ceil(size / workgroupSize);
    
    await this.gpu.dispatch(pipeline, bindGroup, [numWorkgroups, 1, 1]);
    
    return {
      data: new Float32Array(size),
      shape: gradOutput.shape,
      strides: gradOutput.strides,
      device: 'gpu',
      gpuBuffer: outputBuffer
    };
  }
}
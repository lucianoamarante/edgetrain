(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
    typeof define === 'function' && define.amd ? define(['exports'], factory) :
    (global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global.EdgeTrain = {}));
})(this, (function (exports) { 'use strict';

    class WebGPUContext {
        static instance;
        device = null;
        adapter = null;
        constructor() { }
        static getInstance() {
            if (!WebGPUContext.instance) {
                WebGPUContext.instance = new WebGPUContext();
            }
            return WebGPUContext.instance;
        }
        async initialize() {
            if (!navigator.gpu) {
                throw new Error('WebGPU not supported in this browser');
            }
            this.adapter = await navigator.gpu.requestAdapter({
                powerPreference: 'high-performance'
            });
            if (!this.adapter) {
                throw new Error('No appropriate GPUAdapter found');
            }
            this.device = await this.adapter.requestDevice({
                requiredFeatures: ['shader-f16'],
                requiredLimits: {
                    maxStorageBufferBindingSize: 1024 * 1024 * 1024,
                    maxBufferSize: 1024 * 1024 * 1024,
                    maxComputeWorkgroupSizeX: 512,
                    maxComputeWorkgroupSizeY: 512,
                    maxComputeWorkgroupsPerDimension: 65535
                }
            });
            this.device.lost.then((info) => {
                console.error(`WebGPU device was lost: ${info.reason}`, info.message);
                this.device = null;
            });
        }
        getDevice() {
            if (!this.device) {
                throw new Error('WebGPU device not initialized');
            }
            return this.device;
        }
        async createBuffer(size, usage) {
            const device = this.getDevice();
            return device.createBuffer({
                size,
                usage,
                mappedAtCreation: false
            });
        }
        async writeBuffer(buffer, data) {
            const device = this.getDevice();
            device.queue.writeBuffer(buffer, 0, data);
        }
        async readBuffer(buffer, size) {
            const device = this.getDevice();
            const stagingBuffer = device.createBuffer({
                size,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            const commandEncoder = device.createCommandEncoder();
            commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
            device.queue.submit([commandEncoder.finish()]);
            await stagingBuffer.mapAsync(GPUMapMode.READ);
            const copyArrayBuffer = stagingBuffer.getMappedRange();
            const data = new Float32Array(copyArrayBuffer.slice(0));
            stagingBuffer.unmap();
            stagingBuffer.destroy();
            return data;
        }
        createBindGroupLayout(entries) {
            const device = this.getDevice();
            return device.createBindGroupLayout({ entries });
        }
        createComputePipeline(shader, bindGroupLayout, entryPoint = 'main') {
            const device = this.getDevice();
            const shaderModule = device.createShaderModule({
                code: shader
            });
            return device.createComputePipeline({
                layout: device.createPipelineLayout({
                    bindGroupLayouts: [bindGroupLayout]
                }),
                compute: {
                    module: shaderModule,
                    entryPoint
                }
            });
        }
        async dispatch(pipeline, bindGroup, workgroups) {
            const device = this.getDevice();
            const commandEncoder = device.createCommandEncoder();
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(pipeline);
            computePass.setBindGroup(0, bindGroup);
            computePass.dispatchWorkgroups(...workgroups);
            computePass.end();
            device.queue.submit([commandEncoder.finish()]);
            await device.queue.onSubmittedWorkDone();
        }
    }

    class TensorOps {
        static gpu = WebGPUContext.getInstance();
        static async create(data, shape, device = 'gpu') {
            const flatData = data instanceof Float32Array ? data : new Float32Array(data);
            const strides = this.computeStrides(shape);
            let gpuBuffer;
            if (device === 'gpu') {
                await this.gpu.initialize();
                const bufferSize = flatData.byteLength;
                gpuBuffer = await this.gpu.createBuffer(bufferSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
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
        static zeros(shape, device = 'gpu') {
            const size = shape.reduce((a, b) => a * b, 1);
            return this.create(new Float32Array(size), shape, device);
        }
        static ones(shape, device = 'gpu') {
            const size = shape.reduce((a, b) => a * b, 1);
            const data = new Float32Array(size).fill(1);
            return this.create(data, shape, device);
        }
        static random(shape, device = 'gpu') {
            const size = shape.reduce((a, b) => a * b, 1);
            const data = new Float32Array(size);
            for (let i = 0; i < size; i++) {
                data[i] = (Math.random() - 0.5) * 2 * Math.sqrt(6 / (shape[0] + shape[1]));
            }
            return this.create(data, shape, device);
        }
        static async toGPU(tensor) {
            if (tensor.device === 'gpu')
                return tensor;
            await this.gpu.initialize();
            const bufferSize = tensor.data.byteLength;
            const gpuBuffer = await this.gpu.createBuffer(bufferSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
            await this.gpu.writeBuffer(gpuBuffer, tensor.data);
            return {
                ...tensor,
                device: 'gpu',
                gpuBuffer
            };
        }
        static async toCPU(tensor) {
            if (tensor.device === 'cpu')
                return tensor;
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
        static reshape(tensor, newShape) {
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
        static slice(tensor, start, end) {
            if (tensor.device === 'gpu') {
                throw new Error('GPU slicing not yet implemented');
            }
            const newShape = start.map((s, i) => end[i] - s);
            const newSize = newShape.reduce((a, b) => a * b, 1);
            const newData = new Float32Array(newSize);
            const copySlice = (dim, srcOffset, dstOffset) => {
                if (dim === tensor.shape.length) {
                    newData[dstOffset] = tensor.data[srcOffset];
                    return;
                }
                for (let i = start[dim]; i < end[dim]; i++) {
                    copySlice(dim + 1, srcOffset + i * tensor.strides[dim], dstOffset + (i - start[dim]) * this.computeStrides(newShape)[dim]);
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
        static computeStrides(shape) {
            const strides = new Array(shape.length);
            let stride = 1;
            for (let i = shape.length - 1; i >= 0; i--) {
                strides[i] = stride;
                stride *= shape[i];
            }
            return strides;
        }
        static async add(a, b) {
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
            throw new Error('GPU add not yet implemented');
        }
        static async multiply(a, b) {
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
            throw new Error('GPU multiply not yet implemented');
        }
    }

    const matmulShader = `
struct MatmulParams {
  M: u32,
  N: u32,
  K: u32,
  alpha: f32,
  beta: f32,
}

@group(0) @binding(0) var<uniform> params: MatmulParams;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

const TILE_SIZE = 16u;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let row = global_id.x;
  let col = global_id.y;
  
  if (row >= params.M || col >= params.N) {
    return;
  }
  
  var sum = 0.0;
  
  for (var k = 0u; k < params.K; k = k + 1u) {
    let a_index = row * params.K + k;
    let b_index = k * params.N + col;
    sum = sum + a[a_index] * b[b_index];
  }
  
  let c_index = row * params.N + col;
  c[c_index] = params.alpha * sum + params.beta * c[c_index];
}

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn matmul_tiled(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  var tile_a: array<f32, 256>;
  var tile_b: array<f32, 256>;
  
  let row = workgroup_id.x * TILE_SIZE + local_id.x;
  let col = workgroup_id.y * TILE_SIZE + local_id.y;
  
  var sum = 0.0;
  
  let num_tiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;
  
  for (var t = 0u; t < num_tiles; t = t + 1u) {
    let tile_row = row;
    let tile_col = t * TILE_SIZE + local_id.y;
    
    if (tile_row < params.M && tile_col < params.K) {
      tile_a[local_id.x * TILE_SIZE + local_id.y] = a[tile_row * params.K + tile_col];
    } else {
      tile_a[local_id.x * TILE_SIZE + local_id.y] = 0.0;
    }
    
    let b_row = t * TILE_SIZE + local_id.x;
    let b_col = col;
    
    if (b_row < params.K && b_col < params.N) {
      tile_b[local_id.x * TILE_SIZE + local_id.y] = b[b_row * params.N + b_col];
    } else {
      tile_b[local_id.x * TILE_SIZE + local_id.y] = 0.0;
    }
    
    workgroupBarrier();
    
    for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
      sum = sum + tile_a[local_id.x * TILE_SIZE + k] * tile_b[k * TILE_SIZE + local_id.y];
    }
    
    workgroupBarrier();
  }
  
  if (row < params.M && col < params.N) {
    let c_index = row * params.N + col;
    c[c_index] = params.alpha * sum + params.beta * c[c_index];
  }
}
`;
    class MatMulKernel {
        static gpu = WebGPUContext.getInstance();
        static pipelineCache = new Map();
        static async forward(a, b, transposeB = false) {
            if (a.shape.length !== 2 || b.shape.length !== 2) {
                throw new Error('MatMul requires 2D tensors');
            }
            const M = a.shape[0];
            const K = a.shape[1];
            const N = transposeB ? b.shape[0] : b.shape[1];
            if (K !== (transposeB ? b.shape[1] : b.shape[0])) {
                throw new Error('Invalid matrix dimensions for multiplication');
            }
            if (a.device === 'cpu' || b.device === 'cpu') {
                return this.cpuMatMul(a, b, transposeB);
            }
            return this.gpuMatMul(a, b, M, N, K, transposeB);
        }
        static async cpuMatMul(a, b, transposeB) {
            const M = a.shape[0];
            const K = a.shape[1];
            const N = transposeB ? b.shape[0] : b.shape[1];
            const result = new Float32Array(M * N);
            for (let i = 0; i < M; i++) {
                for (let j = 0; j < N; j++) {
                    let sum = 0;
                    for (let k = 0; k < K; k++) {
                        const aIdx = i * K + k;
                        const bIdx = transposeB ? j * K + k : k * N + j;
                        sum += a.data[aIdx] * b.data[bIdx];
                    }
                    result[i * N + j] = sum;
                }
            }
            return TensorOps.create(result, [M, N], 'cpu');
        }
        static async gpuMatMul(a, b, M, N, K, transposeB) {
            const device = this.gpu.getDevice();
            const cacheKey = `matmul_${M}_${N}_${K}_${transposeB}`;
            let pipeline = this.pipelineCache.get(cacheKey);
            if (!pipeline) {
                const bindGroupLayout = this.gpu.createBindGroupLayout([
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'uniform' }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'read-only-storage' }
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'read-only-storage' }
                    },
                    {
                        binding: 3,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'storage' }
                    }
                ]);
                pipeline = this.gpu.createComputePipeline(matmulShader, bindGroupLayout, M * N > 1024 ? 'matmul_tiled' : 'main');
                this.pipelineCache.set(cacheKey, pipeline);
            }
            const paramsBuffer = await this.gpu.createBuffer(20, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
            const params = new Uint32Array([M, N, K, 0]);
            const paramsFloat = new Float32Array(params.buffer);
            paramsFloat[3] = 1.0;
            paramsFloat[4] = 0.0;
            await this.gpu.writeBuffer(paramsBuffer, paramsFloat);
            const resultBuffer = await this.gpu.createBuffer(M * N * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
            const bindGroup = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: paramsBuffer } },
                    { binding: 1, resource: { buffer: a.gpuBuffer } },
                    { binding: 2, resource: { buffer: b.gpuBuffer } },
                    { binding: 3, resource: { buffer: resultBuffer } }
                ]
            });
            const workgroupSize = 16;
            const workgroupsX = Math.ceil(M / workgroupSize);
            const workgroupsY = Math.ceil(N / workgroupSize);
            await this.gpu.dispatch(pipeline, bindGroup, [workgroupsX, workgroupsY, 1]);
            return {
                data: new Float32Array(M * N),
                shape: [M, N],
                strides: [N, 1],
                device: 'gpu',
                gpuBuffer: resultBuffer
            };
        }
    }

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
    class ActivationKernel {
        static gpu = WebGPUContext.getInstance();
        static pipelineCache = new Map();
        static async forward(input, activation) {
            if (input.device === 'cpu') {
                return this.cpuActivation(input, activation);
            }
            return this.gpuActivation(input, activation);
        }
        static async backward(gradOutput, output, activation) {
            if (gradOutput.device === 'cpu') {
                return this.cpuActivationBackward(gradOutput, output, activation);
            }
            return this.gpuActivationBackward(gradOutput, output, activation);
        }
        static async cpuActivation(input, activation) {
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
        static async cpuActivationBackward(gradOutput, output, activation) {
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
                    throw new Error('Softmax backward requires loss integration');
            }
            return TensorOps.create(result, gradOutput.shape, 'cpu');
        }
        static async gpuActivation(input, activation) {
            const device = this.gpu.getDevice();
            const size = input.data.length;
            const cacheKey = `activation_${activation}_${size}`;
            let pipeline = this.pipelineCache.get(cacheKey);
            if (!pipeline) {
                const bindGroupLayout = this.gpu.createBindGroupLayout([
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'read-only-storage' }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'storage' }
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'uniform' }
                    }
                ]);
                pipeline = this.gpu.createComputePipeline(activationShader, bindGroupLayout, activation === 'tanh' ? 'tanh_activation' : activation);
                this.pipelineCache.set(cacheKey, pipeline);
            }
            const sizeBuffer = await this.gpu.createBuffer(4, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
            const sizeArray = new Uint32Array([size]);
            await this.gpu.writeBuffer(sizeBuffer, new Float32Array(sizeArray.buffer));
            const outputBuffer = await this.gpu.createBuffer(size * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
            const bindGroup = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: input.gpuBuffer } },
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
        static async gpuActivationBackward(gradOutput, output, activation) {
            throw new Error('GPU activation backward not yet implemented');
        }
    }

    class DenseLayer {
        weights;
        bias;
        activation;
        lastInput = null;
        lastPreActivation = null;
        lastOutput = null;
        constructor(inputSize, outputSize, activation = null, device = 'gpu') {
            this.activation = activation;
            this.initializeWeights(inputSize, outputSize, device);
        }
        async initializeWeights(inputSize, outputSize, device) {
            const scale = Math.sqrt(2.0 / inputSize);
            const weightData = new Float32Array(inputSize * outputSize);
            for (let i = 0; i < weightData.length; i++) {
                weightData[i] = (Math.random() - 0.5) * 2 * scale;
            }
            const biasData = new Float32Array(outputSize).fill(0);
            this.weights = await TensorOps.create(weightData, [inputSize, outputSize], device);
            this.bias = await TensorOps.create(biasData, [outputSize], device);
        }
        async forward(input) {
            this.lastInput = input;
            let output = await MatMulKernel.forward(input, this.weights);
            if (this.bias.device !== output.device) {
                const biasOnDevice = this.bias.device === 'cpu'
                    ? await TensorOps.toGPU(this.bias)
                    : await TensorOps.toCPU(this.bias);
                output = await this.addBias(output, biasOnDevice);
            }
            else {
                output = await this.addBias(output, this.bias);
            }
            this.lastPreActivation = output;
            if (this.activation) {
                output = await ActivationKernel.forward(output, this.activation);
            }
            this.lastOutput = output;
            return output;
        }
        async backward(gradOutput) {
            if (!this.lastInput || !this.lastPreActivation || !this.lastOutput) {
                throw new Error('Forward pass must be called before backward pass');
            }
            let gradPreActivation = gradOutput;
            if (this.activation) {
                gradPreActivation = await ActivationKernel.backward(gradOutput, this.lastOutput, this.activation);
            }
            const gradInput = await MatMulKernel.forward(gradPreActivation, this.weights, true);
            const gradWeights = await MatMulKernel.forward(this.lastInput, gradPreActivation, false);
            const gradBias = await this.sumGradients(gradPreActivation);
            return { gradInput, gradWeights, gradBias };
        }
        async addBias(input, bias) {
            if (input.device === 'cpu') {
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
            throw new Error('GPU bias addition not yet implemented');
        }
        async sumGradients(gradients) {
            if (gradients.device === 'cpu') {
                const batchSize = gradients.shape[0];
                const outputSize = gradients.shape[1];
                const result = new Float32Array(outputSize).fill(0);
                for (let b = 0; b < batchSize; b++) {
                    for (let o = 0; o < outputSize; o++) {
                        result[o] += gradients.data[b * outputSize + o];
                    }
                }
                return TensorOps.create(result, [outputSize], 'cpu');
            }
            throw new Error('GPU gradient summation not yet implemented');
        }
        getWeights() {
            return this.weights;
        }
        getBias() {
            return this.bias;
        }
        async updateWeights(gradWeights, gradBias, learningRate) {
            if (this.weights.device === 'cpu') {
                for (let i = 0; i < this.weights.data.length; i++) {
                    this.weights.data[i] -= learningRate * gradWeights.data[i];
                }
                for (let i = 0; i < this.bias.data.length; i++) {
                    this.bias.data[i] -= learningRate * gradBias.data[i];
                }
            }
            else {
                throw new Error('GPU weight updates not yet implemented');
            }
        }
    }

    class Model {
        layers = [];
        config;
        constructor(config) {
            this.config = config;
        }
        addDenseLayer(inputSize, outputSize, activation) {
            const layer = new DenseLayer(inputSize, outputSize, activation, this.config.device);
            this.layers.push(layer);
        }
        async forward(input) {
            let output = input;
            for (const layer of this.layers) {
                output = await layer.forward(output);
            }
            return output;
        }
        async backward(loss) {
            let gradOutput = loss;
            for (let i = this.layers.length - 1; i >= 0; i--) {
                const { gradInput, gradWeights, gradBias } = await this.layers[i].backward(gradOutput);
                await this.layers[i].updateWeights(gradWeights, gradBias, this.config.learningRate);
                gradOutput = gradInput;
            }
        }
        async train(inputs, targets, epochs, onEpochEnd) {
            const batchSize = this.config.batchSize;
            const numSamples = inputs.shape[0];
            const numBatches = Math.ceil(numSamples / batchSize);
            for (let epoch = 0; epoch < epochs; epoch++) {
                let totalLoss = 0;
                for (let batch = 0; batch < numBatches; batch++) {
                    const startIdx = batch * batchSize;
                    const endIdx = Math.min(startIdx + batchSize, numSamples);
                    const batchInputs = await this.sliceBatch(inputs, startIdx, endIdx);
                    const batchTargets = await this.sliceBatch(targets, startIdx, endIdx);
                    const predictions = await this.forward(batchInputs);
                    const loss = await this.computeLoss(predictions, batchTargets);
                    const lossGradient = await this.computeLossGradient(predictions, batchTargets);
                    await this.backward(lossGradient);
                    totalLoss += await this.tensorSum(loss);
                }
                const avgLoss = totalLoss / numBatches;
                if (onEpochEnd) {
                    onEpochEnd(epoch, avgLoss);
                }
            }
        }
        async predict(input) {
            return this.forward(input);
        }
        async sliceBatch(tensor, start, end) {
            if (tensor.device === 'gpu') {
                throw new Error('GPU batch slicing not yet implemented');
            }
            const batchSize = end - start;
            const featureSize = tensor.shape.slice(1).reduce((a, b) => a * b, 1);
            const batchData = new Float32Array(batchSize * featureSize);
            for (let i = 0; i < batchSize; i++) {
                const sourceStart = (start + i) * featureSize;
                const targetStart = i * featureSize;
                for (let j = 0; j < featureSize; j++) {
                    batchData[targetStart + j] = tensor.data[sourceStart + j];
                }
            }
            return TensorOps.create(batchData, [batchSize, ...tensor.shape.slice(1)], tensor.device);
        }
        async computeLoss(predictions, targets) {
            if (predictions.device === 'cpu') {
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
            throw new Error('GPU loss computation not yet implemented');
        }
        async computeLossGradient(predictions, targets) {
            if (predictions.device === 'cpu') {
                const gradients = new Float32Array(predictions.data.length);
                for (let i = 0; i < predictions.data.length; i++) {
                    Math.max(predictions.data[i], 1e-15);
                    gradients[i] = predictions.data[i] - targets.data[i];
                }
                return TensorOps.create(gradients, predictions.shape, 'cpu');
            }
            throw new Error('GPU loss gradient computation not yet implemented');
        }
        async tensorSum(tensor) {
            if (tensor.device === 'cpu') {
                return tensor.data.reduce((sum, val) => sum + val, 0);
            }
            throw new Error('GPU tensor sum not yet implemented');
        }
    }

    class MNISTDemo {
        model;
        constructor(device = 'gpu') {
            this.model = new Model({
                learningRate: 0.001,
                batchSize: 32,
                device
            });
            this.model.addDenseLayer(784, 128, 'relu');
            this.model.addDenseLayer(128, 64, 'relu');
            this.model.addDenseLayer(64, 10, 'softmax');
        }
        async generateSyntheticData(numSamples) {
            const inputData = new Float32Array(numSamples * 784);
            const labelData = new Float32Array(numSamples * 10);
            for (let i = 0; i < numSamples; i++) {
                for (let j = 0; j < 784; j++) {
                    inputData[i * 784 + j] = Math.random();
                }
                const label = Math.floor(Math.random() * 10);
                labelData[i * 10 + label] = 1.0;
            }
            const inputs = await TensorOps.create(inputData, [numSamples, 784], this.model['config'].device);
            const labels = await TensorOps.create(labelData, [numSamples, 10], this.model['config'].device);
            return { inputs, labels };
        }
        async train(numSamples = 1000, epochs = 10) {
            console.log('Generating synthetic MNIST-like data...');
            const { inputs, labels } = await this.generateSyntheticData(numSamples);
            console.log('Starting training...');
            await this.model.train(inputs, labels, epochs, (epoch, loss) => {
                console.log(`Epoch ${epoch + 1}/${epochs}, Loss: ${loss.toFixed(4)}`);
            });
            console.log('Training completed!');
        }
        async predict(input) {
            if (input.length !== 784) {
                throw new Error('Input must be 784 pixels (28x28)');
            }
            const inputTensor = await TensorOps.create(new Float32Array(input), [1, 784], this.model['config'].device);
            const prediction = await this.model.predict(inputTensor);
            if (prediction.device === 'gpu') {
                const cpuPrediction = await TensorOps.toCPU(prediction);
                return Array.from(cpuPrediction.data);
            }
            return Array.from(prediction.data);
        }
        async benchmark() {
            const numSamples = 100;
            const epochs = 5;
            const startTraining = performance.now();
            await this.train(numSamples, epochs);
            const trainingTime = performance.now() - startTraining;
            const { inputs, labels } = await this.generateSyntheticData(20);
            const startInference = performance.now();
            let correct = 0;
            for (let i = 0; i < 20; i++) {
                const sampleInput = Array.from(inputs.data.slice(i * 784, (i + 1) * 784));
                const prediction = await this.predict(sampleInput);
                const predictedClass = prediction.indexOf(Math.max(...prediction));
                const actualClass = Array.from(labels.data.slice(i * 10, (i + 1) * 10)).indexOf(1);
                if (predictedClass === actualClass) {
                    correct++;
                }
            }
            const inferenceTime = performance.now() - startInference;
            const accuracy = correct / 20;
            return {
                trainingTime,
                inferenceTime,
                accuracy
            };
        }
    }

    class EdgeTrain {
        static async initialize() {
            try {
                const gpu = WebGPUContext.getInstance();
                await gpu.initialize();
                return true;
            }
            catch (error) {
                console.error('EdgeTrain initialization failed:', error);
                return false;
            }
        }
        static createModel(config) {
            return new Model(config);
        }
        static async createDemo(device = 'gpu') {
            if (device === 'gpu') {
                const initialized = await EdgeTrain.initialize();
                if (!initialized) {
                    console.warn('WebGPU initialization failed, falling back to CPU');
                    device = 'cpu';
                }
            }
            return new MNISTDemo(device);
        }
        static async isWebGPUSupported() {
            return 'gpu' in navigator;
        }
        static async getDeviceInfo() {
            const webgpuSupported = await EdgeTrain.isWebGPUSupported();
            if (!webgpuSupported) {
                return {
                    webgpuSupported: false,
                    device: 'CPU fallback',
                    features: []
                };
            }
            try {
                const adapter = await navigator.gpu.requestAdapter();
                if (!adapter) {
                    return {
                        webgpuSupported: false,
                        device: 'No adapter available',
                        features: []
                    };
                }
                return {
                    webgpuSupported: true,
                    device: adapter.info?.device || 'Unknown GPU',
                    features: Array.from(adapter.features)
                };
            }
            catch (error) {
                return {
                    webgpuSupported: false,
                    device: 'GPU initialization failed',
                    features: []
                };
            }
        }
    }

    exports.ActivationKernel = ActivationKernel;
    exports.DenseLayer = DenseLayer;
    exports.EdgeTrain = EdgeTrain;
    exports.MNISTDemo = MNISTDemo;
    exports.MatMulKernel = MatMulKernel;
    exports.Model = Model;
    exports.TensorOps = TensorOps;
    exports.WebGPUContext = WebGPUContext;

}));
//# sourceMappingURL=index.umd.js.map

import { Tensor } from '../core/types';
import { WebGPUContext } from '../core/webgpu';
import { TensorOps } from '../core/tensor';
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

export class MatMulKernel {
  private static gpu = WebGPUContext.getInstance();
  private static pipelineCache = new Map<string, GPUComputePipeline>();
  
  static async forward(a: Tensor, b: Tensor, transposeB = false): Promise<Tensor> {
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
  
  private static async cpuMatMul(a: Tensor, b: Tensor, transposeB: boolean): Promise<Tensor> {
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
  
  private static async gpuMatMul(
    a: Tensor,
    b: Tensor,
    M: number,
    N: number,
    K: number,
    transposeB: boolean
  ): Promise<Tensor> {
    const device = this.gpu.getDevice();
    
    const cacheKey = `matmul_${M}_${N}_${K}_${transposeB}`;
    let pipeline = this.pipelineCache.get(cacheKey);
    
    if (!pipeline) {
      const bindGroupLayout = this.gpu.createBindGroupLayout([
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' as GPUBufferBindingType }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' as GPUBufferBindingType }
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' as GPUBufferBindingType }
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' as GPUBufferBindingType }
        }
      ]);
      
      pipeline = this.gpu.createComputePipeline(
        matmulShader,
        bindGroupLayout,
        M * N > 1024 ? 'matmul_tiled' : 'main'
      );
      
      this.pipelineCache.set(cacheKey, pipeline);
    }
    
    const paramsBuffer = await this.gpu.createBuffer(
      20,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    
    const params = new Uint32Array([M, N, K, 0]);
    const paramsFloat = new Float32Array(params.buffer);
    paramsFloat[3] = 1.0;
    paramsFloat[4] = 0.0;
    
    await this.gpu.writeBuffer(paramsBuffer, paramsFloat);
    
    const resultBuffer = await this.gpu.createBuffer(
      M * N * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: a.gpuBuffer! } },
        { binding: 2, resource: { buffer: b.gpuBuffer! } },
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
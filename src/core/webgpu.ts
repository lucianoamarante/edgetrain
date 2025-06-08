export class WebGPUContext {
  private static instance: WebGPUContext;
  private device: GPUDevice | null = null;
  private adapter: GPUAdapter | null = null;
  
  private constructor() {}
  
  static getInstance(): WebGPUContext {
    if (!WebGPUContext.instance) {
      WebGPUContext.instance = new WebGPUContext();
    }
    return WebGPUContext.instance;
  }
  
  async initialize(): Promise<void> {
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported in this browser');
    }
    
    this.adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance'
    });
    
    if (!this.adapter) {
      throw new Error('No appropriate GPUAdapter found');
    }
    
    try {
      this.device = await this.adapter.requestDevice({
        requiredLimits: {
          maxStorageBufferBindingSize: 1024 * 1024 * 1024,
          maxBufferSize: 1024 * 1024 * 1024,
          maxComputeWorkgroupSizeX: 512,
          maxComputeWorkgroupSizeY: 512,
          maxComputeWorkgroupsPerDimension: 65535
        }
      });
    } catch (error) {
      // Fallback to basic device if advanced features not available
      this.device = await this.adapter.requestDevice();
    }
    
    this.device.lost.then((info) => {
      console.error(`WebGPU device was lost: ${info.reason}`, info.message);
      this.device = null;
    });
  }
  
  getDevice(): GPUDevice {
    if (!this.device) {
      throw new Error('WebGPU device not initialized');
    }
    return this.device;
  }
  
  async createBuffer(size: number, usage: GPUBufferUsageFlags): Promise<GPUBuffer> {
    const device = this.getDevice();
    return device.createBuffer({
      size,
      usage,
      mappedAtCreation: false
    });
  }
  
  async writeBuffer(buffer: GPUBuffer, data: Float32Array): Promise<void> {
    const device = this.getDevice();
    device.queue.writeBuffer(buffer, 0, data);
  }
  
  async readBuffer(buffer: GPUBuffer, size: number): Promise<Float32Array> {
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
  
  createBindGroupLayout(entries: GPUBindGroupLayoutEntry[]): GPUBindGroupLayout {
    const device = this.getDevice();
    return device.createBindGroupLayout({ entries });
  }
  
  createComputePipeline(
    shader: string,
    bindGroupLayout: GPUBindGroupLayout,
    entryPoint: string = 'main'
  ): GPUComputePipeline {
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
  
  async dispatch(
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroups: [number, number, number]
  ): Promise<void> {
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
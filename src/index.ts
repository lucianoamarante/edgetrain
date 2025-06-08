export type { Tensor } from './core/types';
export { TensorOps } from './core/tensor';
export { WebGPUContext } from './core/webgpu';
export { Model } from './core/model';
export type { ModelConfig } from './core/model';
export { DenseLayer } from './layers/dense';
export { MatMulKernel } from './kernels/matmul';
export { ActivationKernel } from './kernels/activation';
export type { ActivationType } from './kernels/activation';
export { MNISTDemo } from './demo/mnist';

import { WebGPUContext } from './core/webgpu';
import { Model, ModelConfig } from './core/model';
import { MNISTDemo } from './demo/mnist';

export class EdgeTrain {
  static async initialize(): Promise<boolean> {
    try {
      const gpu = WebGPUContext.getInstance();
      await gpu.initialize();
      return true;
    } catch (error) {
      console.error('EdgeTrain initialization failed:', error);
      return false;
    }
  }
  
  static createModel(config: ModelConfig): Model {
    return new Model(config);
  }
  
  static async createDemo(device: 'cpu' | 'gpu' = 'gpu'): Promise<MNISTDemo> {
    if (device === 'gpu') {
      const initialized = await EdgeTrain.initialize();
      if (!initialized) {
        console.warn('WebGPU initialization failed, falling back to CPU');
        device = 'cpu';
      }
    }
    
    return new MNISTDemo(device);
  }
  
  static async isWebGPUSupported(): Promise<boolean> {
    return 'gpu' in navigator;
  }
  
  static async getDeviceInfo(): Promise<{
    webgpuSupported: boolean;
    device: string;
    features: string[];
  }> {
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
    } catch (error) {
      return {
        webgpuSupported: false,
        device: 'GPU initialization failed',
        features: []
      };
    }
  }
}
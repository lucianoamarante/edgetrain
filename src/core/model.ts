import { Tensor } from './types';
import { TensorOps } from './tensor';
import { DenseLayer } from '../layers/dense';

export interface ModelConfig {
  learningRate: number;
  batchSize: number;
  device: 'cpu' | 'gpu';
}

export class Model {
  private layers: DenseLayer[] = [];
  private config: ModelConfig;
  
  constructor(config: ModelConfig) {
    this.config = config;
  }
  
  addDenseLayer(
    inputSize: number,
    outputSize: number,
    activation?: 'relu' | 'sigmoid' | 'tanh' | 'softmax'
  ): void {
    const layer = new DenseLayer(inputSize, outputSize, activation, this.config.device);
    this.layers.push(layer);
  }
  
  async forward(input: Tensor): Promise<Tensor> {
    let output = input;
    
    for (const layer of this.layers) {
      output = await layer.forward(output);
    }
    
    return output;
  }
  
  async backward(loss: Tensor): Promise<void> {
    let gradOutput = loss;
    
    for (let i = this.layers.length - 1; i >= 0; i--) {
      const { gradInput, gradWeights, gradBias } = await this.layers[i].backward(gradOutput);
      
      await this.layers[i].updateWeights(gradWeights, gradBias, this.config.learningRate);
      
      gradOutput = gradInput;
    }
  }
  
  async train(
    inputs: Tensor,
    targets: Tensor,
    epochs: number,
    onEpochEnd?: (epoch: number, loss: number) => void
  ): Promise<void> {
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
  
  async predict(input: Tensor): Promise<Tensor> {
    return this.forward(input);
  }
  
  private async sliceBatch(tensor: Tensor, start: number, end: number): Promise<Tensor> {
    if (tensor.device === 'gpu') {
      const cpuTensor = await TensorOps.toCPU(tensor);
      const result = await this.sliceBatch(cpuTensor, start, end);
      return TensorOps.toGPU(result);
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
  
  private async computeLoss(predictions: Tensor, targets: Tensor): Promise<Tensor> {
    const { LossKernel } = await import('../kernels/loss');
    return LossKernel.crossEntropyLoss(predictions, targets);
  }
  
  private async computeLossGradient(predictions: Tensor, targets: Tensor): Promise<Tensor> {
    const { LossKernel } = await import('../kernels/loss');
    return LossKernel.crossEntropyGradient(predictions, targets);
  }
  
  private async tensorSum(tensor: Tensor): Promise<number> {
    const { LossKernel } = await import('../kernels/loss');
    return LossKernel.tensorSum(tensor);
  }
}
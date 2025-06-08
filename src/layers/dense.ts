import { Tensor } from '../core/types';
import { TensorOps } from '../core/tensor';
import { MatMulKernel } from '../kernels/matmul';
import { ActivationKernel, ActivationType } from '../kernels/activation';

export class DenseLayer {
  private weights: Tensor;
  private bias: Tensor;
  private activation: ActivationType | null;
  private device: 'cpu' | 'gpu';
  
  private lastInput: Tensor | null = null;
  private lastPreActivation: Tensor | null = null;
  private lastOutput: Tensor | null = null;
  
  constructor(
    inputSize: number,
    outputSize: number,
    activation: ActivationType | null = null,
    device: 'cpu' | 'gpu' = 'gpu'
  ) {
    this.activation = activation;
    this.device = device;
    const { weights, bias } = this.initializeWeights(inputSize, outputSize);
    this.weights = weights;
    this.bias = bias;
  }
  
  private initializeWeights(
    inputSize: number,
    outputSize: number
  ): { weights: Tensor; bias: Tensor } {
    const scale = Math.sqrt(2.0 / inputSize);
    
    const weightData = new Float32Array(inputSize * outputSize);
    for (let i = 0; i < weightData.length; i++) {
      weightData[i] = (Math.random() - 0.5) * 2 * scale;
    }
    
    const biasData = new Float32Array(outputSize).fill(0);
    
    // Create CPU tensors initially - will convert to GPU when needed
    const weights: Tensor = {
      data: weightData,
      shape: [inputSize, outputSize],
      strides: [outputSize, 1],
      device: 'cpu'
    };
    
    const bias: Tensor = {
      data: biasData,
      shape: [outputSize],
      strides: [1],
      device: 'cpu'
    };
    
    return { weights, bias };
  }
  
  async forward(input: Tensor): Promise<Tensor> {
    this.lastInput = input;
    
    // Ensure weights and bias are on the same device as input
    let weights = this.weights;
    let bias = this.bias;
    
    if (this.device === 'gpu' && input.device === 'gpu') {
      if (weights.device === 'cpu') {
        weights = await TensorOps.toGPU(weights);
        this.weights = weights;
      }
      if (bias.device === 'cpu') {
        bias = await TensorOps.toGPU(bias);
        this.bias = bias;
      }
    }
    
    let output = await MatMulKernel.forward(input, weights);
    output = await this.addBias(output, bias);
    
    this.lastPreActivation = output;
    
    if (this.activation) {
      output = await ActivationKernel.forward(output, this.activation);
    }
    
    this.lastOutput = output;
    return output;
  }
  
  async backward(gradOutput: Tensor): Promise<{
    gradInput: Tensor;
    gradWeights: Tensor;
    gradBias: Tensor;
  }> {
    if (!this.lastInput || !this.lastPreActivation || !this.lastOutput) {
      throw new Error('Forward pass must be called before backward pass');
    }
    
    let gradPreActivation = gradOutput;
    
    if (this.activation) {
      gradPreActivation = await ActivationKernel.backward(
        gradOutput,
        this.lastOutput,
        this.activation
      );
    }
    
    // grad_input = grad_output @ weights^T
    const gradInput = await MatMulKernel.forward(
      gradPreActivation,
      this.weights,
      true
    );
    
    // grad_weights = input^T @ grad_output  
    const inputTransposed = this.transpose(this.lastInput);
    const gradWeights = await MatMulKernel.forward(
      inputTransposed,
      gradPreActivation
    );
    
    const gradBias = await this.sumGradients(gradPreActivation);
    
    return { gradInput, gradWeights, gradBias };
  }
  
  private async addBias(input: Tensor, bias: Tensor): Promise<Tensor> {
    const { ElementwiseKernel } = await import('../kernels/elementwise');
    return ElementwiseKernel.addBias(input, bias);
  }
  
  private async sumGradients(gradients: Tensor): Promise<Tensor> {
    const { ElementwiseKernel } = await import('../kernels/elementwise');
    return ElementwiseKernel.sumGradients(gradients, [gradients.shape[1]]);
  }
  
  getWeights(): Tensor {
    return this.weights;
  }
  
  getBias(): Tensor {
    return this.bias;
  }
  
  private transpose(tensor: Tensor): Tensor {
    if (tensor.shape.length !== 2) {
      throw new Error('Transpose only supports 2D tensors');
    }
    
    const [rows, cols] = tensor.shape;
    const transposed = new Float32Array(rows * cols);
    
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        transposed[j * rows + i] = tensor.data[i * cols + j];
      }
    }
    
    return {
      data: transposed,
      shape: [cols, rows],
      strides: [rows, 1],
      device: tensor.device,
      gpuBuffer: undefined
    };
  }
  
  async updateWeights(gradWeights: Tensor, gradBias: Tensor, learningRate: number): Promise<void> {
    if (this.weights.device === 'cpu') {
      for (let i = 0; i < this.weights.data.length; i++) {
        this.weights.data[i] -= learningRate * gradWeights.data[i];
      }
      
      for (let i = 0; i < this.bias.data.length; i++) {
        this.bias.data[i] -= learningRate * gradBias.data[i];
      }
    } else {
      const { ElementwiseKernel } = await import('../kernels/elementwise');
      
      const lrTensor = await TensorOps.create(new Float32Array([learningRate]), [1], this.weights.device);
      const scaledGradWeights = await ElementwiseKernel.multiply(gradWeights, lrTensor);
      const scaledGradBias = await ElementwiseKernel.multiply(gradBias, lrTensor);
      
      this.weights = await ElementwiseKernel.subtract(this.weights, scaledGradWeights);
      this.bias = await ElementwiseKernel.subtract(this.bias, scaledGradBias);
    }
  }
}
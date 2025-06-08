import { Model } from '../core/model';
import { TensorOps } from '../core/tensor';
import { Tensor } from '../core/types';

export class MNISTDemo {
  private model: Model;
  
  constructor(device: 'cpu' | 'gpu' = 'gpu') {
    this.model = new Model({
      learningRate: 0.001,
      batchSize: 32,
      device
    });
    
    this.model.addDenseLayer(784, 128, 'relu');
    this.model.addDenseLayer(128, 64, 'relu');
    this.model.addDenseLayer(64, 10, 'softmax');
  }
  
  async generateSyntheticData(numSamples: number): Promise<{
    inputs: Tensor;
    labels: Tensor;
  }> {
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
  
  async train(numSamples: number = 1000, epochs: number = 10): Promise<void> {
    console.log('Generating synthetic MNIST-like data...');
    const { inputs, labels } = await this.generateSyntheticData(numSamples);
    
    console.log('Starting training...');
    await this.model.train(inputs, labels, epochs, (epoch, loss) => {
      console.log(`Epoch ${epoch + 1}/${epochs}, Loss: ${loss.toFixed(4)}`);
    });
    
    console.log('Training completed!');
  }
  
  async predict(input: number[]): Promise<number[]> {
    if (input.length !== 784) {
      throw new Error('Input must be 784 pixels (28x28)');
    }
    
    const inputTensor = await TensorOps.create(
      new Float32Array(input),
      [1, 784],
      this.model['config'].device
    );
    
    const prediction = await this.model.predict(inputTensor);
    
    if (prediction.device === 'gpu') {
      const cpuPrediction = await TensorOps.toCPU(prediction);
      return Array.from(cpuPrediction.data);
    }
    
    return Array.from(prediction.data);
  }
  
  async benchmark(): Promise<{
    trainingTime: number;
    inferenceTime: number;
    accuracy: number;
  }> {
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
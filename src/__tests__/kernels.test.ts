import { MatMulKernel } from '../kernels/matmul';
import { ActivationKernel } from '../kernels/activation';
import { ElementwiseKernel } from '../kernels/elementwise';
import { LossKernel } from '../kernels/loss';
import { TensorOps } from '../core/tensor';

describe('Kernels', () => {
  describe('MatMulKernel', () => {
    it('should perform CPU matrix multiplication', async () => {
      const a = await TensorOps.create(
        [1, 2, 3, 4], [2, 2], 'cpu'
      );
      
      const b = await TensorOps.create(
        [5, 6, 7, 8], [2, 2], 'cpu'
      );
      
      const result = await MatMulKernel.forward(a, b);
      
      expect(result.shape).toEqual([2, 2]);
      expect(result.data).toEqual(new Float32Array([19, 22, 43, 50]));
    });
    
    it('should handle transposed multiplication', async () => {
      const a = await TensorOps.create(
        [1, 2, 3, 4], [2, 2], 'cpu'
      );
      
      const b = await TensorOps.create(
        [5, 7, 6, 8], [2, 2], 'cpu'
      );
      
      const result = await MatMulKernel.forward(a, b, true);
      
      expect(result.shape).toEqual([2, 2]);
      expect(result.data).toEqual(new Float32Array([19, 22, 43, 50]));
    });
    
    it('should validate matrix dimensions', async () => {
      const a = await TensorOps.create([1, 2], [1, 2], 'cpu');
      const b = await TensorOps.create([1, 2, 3], [3, 1], 'cpu');
      
      await expect(MatMulKernel.forward(a, b)).rejects.toThrow();
    });
  });
  
  describe('ActivationKernel', () => {
    it('should apply ReLU activation', async () => {
      const input = await TensorOps.create([-2, -1, 0, 1, 2], [5], 'cpu');
      
      const result = await ActivationKernel.forward(input, 'relu');
      
      expect(result.data).toEqual(new Float32Array([0, 0, 0, 1, 2]));
    });
    
    it('should apply sigmoid activation', async () => {
      const input = await TensorOps.create([0, 1, -1], [3], 'cpu');
      
      const result = await ActivationKernel.forward(input, 'sigmoid');
      
      expect(result.data[0]).toBeCloseTo(0.5, 3);
      expect(result.data[1]).toBeCloseTo(0.731, 3);
      expect(result.data[2]).toBeCloseTo(0.269, 3);
    });
    
    it('should apply tanh activation', async () => {
      const input = await TensorOps.create([0, 1, -1], [3], 'cpu');
      
      const result = await ActivationKernel.forward(input, 'tanh');
      
      expect(result.data[0]).toBeCloseTo(0, 3);
      expect(result.data[1]).toBeCloseTo(0.762, 3);
      expect(result.data[2]).toBeCloseTo(-0.762, 3);
    });
    
    it('should apply softmax activation', async () => {
      const input = await TensorOps.create(
        [1, 2, 3, 1, 1, 1], [2, 3], 'cpu'
      );
      
      const result = await ActivationKernel.forward(input, 'softmax');
      
      // Each row should sum to 1
      const sum1 = result.data[0] + result.data[1] + result.data[2];
      const sum2 = result.data[3] + result.data[4] + result.data[5];
      
      expect(sum1).toBeCloseTo(1, 3);
      expect(sum2).toBeCloseTo(1, 3);
      
      // Second row should have equal probabilities
      expect(result.data[3]).toBeCloseTo(0.333, 3);
      expect(result.data[4]).toBeCloseTo(0.333, 3);
      expect(result.data[5]).toBeCloseTo(0.333, 3);
    });
    
    it('should compute ReLU backward pass', async () => {
      const gradOutput = await TensorOps.ones([3], 'cpu');
      const output = await TensorOps.create([0, 1, 2], [3], 'cpu');
      
      const result = await ActivationKernel.backward(gradOutput, output, 'relu');
      
      expect(result.data).toEqual(new Float32Array([0, 1, 1]));
    });
    
    it('should compute sigmoid backward pass', async () => {
      const gradOutput = await TensorOps.ones([2], 'cpu');
      const output = await TensorOps.create([0.5, 0.731], [2], 'cpu');
      
      const result = await ActivationKernel.backward(gradOutput, output, 'sigmoid');
      
      expect(result.data[0]).toBeCloseTo(0.25, 3);
      expect(result.data[1]).toBeCloseTo(0.197, 3);
    });
  });
  
  describe('ElementwiseKernel', () => {
    it('should add tensors element-wise', async () => {
      const a = await TensorOps.create([1, 2, 3], [3], 'cpu');
      const b = await TensorOps.create([4, 5, 6], [3], 'cpu');
      
      const result = await ElementwiseKernel.add(a, b);
      
      expect(result.data).toEqual(new Float32Array([5, 7, 9]));
    });
    
    it('should multiply tensors element-wise', async () => {
      const a = await TensorOps.create([1, 2, 3], [3], 'cpu');
      const b = await TensorOps.create([4, 5, 6], [3], 'cpu');
      
      const result = await ElementwiseKernel.multiply(a, b);
      
      expect(result.data).toEqual(new Float32Array([4, 10, 18]));
    });
    
    it('should subtract tensors element-wise', async () => {
      const a = await TensorOps.create([4, 5, 6], [3], 'cpu');
      const b = await TensorOps.create([1, 2, 3], [3], 'cpu');
      
      const result = await ElementwiseKernel.subtract(a, b);
      
      expect(result.data).toEqual(new Float32Array([3, 3, 3]));
    });
    
    it('should add bias correctly', async () => {
      const input = await TensorOps.create(
        [1, 2, 3, 4], [2, 2], 'cpu'
      );
      const bias = await TensorOps.create([0.1, 0.2], [2], 'cpu');
      
      const result = await ElementwiseKernel.addBias(input, bias);
      
      expect(result.data).toEqual(new Float32Array([1.1, 2.2, 3.1, 4.2]));
    });
    
    it('should sum gradients correctly', async () => {
      const gradients = await TensorOps.create(
        [1, 2, 3, 4, 5, 6], [3, 2], 'cpu'
      );
      
      const result = await ElementwiseKernel.sumGradients(gradients, [2]);
      
      expect(result.data).toEqual(new Float32Array([9, 12]));
    });
  });
  
  describe('LossKernel', () => {
    it('should compute cross-entropy loss', async () => {
      const predictions = await TensorOps.create(
        [0.7, 0.3, 0.4, 0.6], [2, 2], 'cpu'
      );
      
      const targets = await TensorOps.create(
        [1, 0, 0, 1], [2, 2], 'cpu'
      );
      
      const loss = await LossKernel.crossEntropyLoss(predictions, targets);
      
      expect(loss.shape).toEqual([2]);
      expect(loss.data[0]).toBeCloseTo(-Math.log(0.7), 3);
      expect(loss.data[1]).toBeCloseTo(-Math.log(0.6), 3);
    });
    
    it('should compute cross-entropy gradient', async () => {
      const predictions = await TensorOps.create(
        [0.7, 0.3, 0.4, 0.6], [2, 2], 'cpu'
      );
      
      const targets = await TensorOps.create(
        [1, 0, 0, 1], [2, 2], 'cpu'
      );
      
      const gradient = await LossKernel.crossEntropyGradient(predictions, targets);
      
      expect(gradient.shape).toEqual([2, 2]);
      expect(gradient.data[0]).toBeCloseTo(-0.3, 3);
      expect(gradient.data[1]).toBeCloseTo(0.3, 3);
      expect(gradient.data[2]).toBeCloseTo(0.4, 3);
      expect(gradient.data[3]).toBeCloseTo(-0.4, 3);
    });
    
    it('should compute tensor sum', async () => {
      const tensor = await TensorOps.create([1, 2, 3, 4, 5], [5], 'cpu');
      
      const sum = await LossKernel.tensorSum(tensor);
      
      expect(sum).toBe(15);
    });
    
    it('should handle empty tensor sum', async () => {
      const tensor = await TensorOps.create([], [0], 'cpu');
      
      const sum = await LossKernel.tensorSum(tensor);
      
      expect(sum).toBe(0);
    });
  });
});
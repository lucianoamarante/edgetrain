import { DenseLayer } from '../layers/dense';
import { TensorOps } from '../core/tensor';

describe('DenseLayer', () => {
  describe('construction', () => {
    it('should create layer with correct dimensions', () => {
      const layer = new DenseLayer(10, 5, 'relu', 'cpu');
      
      expect(layer).toBeDefined();
    });
  });
  
  describe('forward pass', () => {
    it('should perform forward pass without activation', async () => {
      const layer = new DenseLayer(3, 2, null, 'cpu');
      const input = await TensorOps.create([1, 2, 3, 4, 5, 6], [2, 3], 'cpu');
      
      const output = await layer.forward(input);
      
      expect(output.shape).toEqual([2, 2]);
      expect(output.device).toBe('cpu');
      expect(output.data.length).toBe(4);
    });
    
    it('should perform forward pass with ReLU activation', async () => {
      const layer = new DenseLayer(3, 2, 'relu', 'cpu');
      const input = await TensorOps.create([1, 2, 3], [1, 3], 'cpu');
      
      const output = await layer.forward(input);
      
      expect(output.shape).toEqual([1, 2]);
      expect(output.device).toBe('cpu');
      
      // ReLU should ensure non-negative outputs
      for (const value of output.data) {
        expect(value).toBeGreaterThanOrEqual(0);
      }
    });
    
    it('should handle batch input correctly', async () => {
      const layer = new DenseLayer(2, 3, null, 'cpu');
      const batchInput = await TensorOps.create(
        [1, 2, 3, 4, 5, 6], [3, 2], 'cpu'
      );
      
      const output = await layer.forward(batchInput);
      
      expect(output.shape).toEqual([3, 3]);
      expect(output.device).toBe('cpu');
    });
  });
  
  describe('backward pass', () => {
    it('should compute gradients correctly', async () => {
      const layer = new DenseLayer(2, 2, null, 'cpu');
      const input = await TensorOps.create([1, 2], [1, 2], 'cpu');
      
      // Forward pass first
      await layer.forward(input);
      
      const gradOutput = await TensorOps.create([1, 1], [1, 2], 'cpu');
      const { gradInput, gradWeights, gradBias } = await layer.backward(gradOutput);
      
      expect(gradInput.shape).toEqual([1, 2]);
      expect(gradWeights.shape).toEqual([2, 2]);
      expect(gradBias.shape).toEqual([2]);
    });
    
    it('should throw error if forward pass not called first', async () => {
      const layer = new DenseLayer(2, 2, null, 'cpu');
      const gradOutput = await TensorOps.create([1, 1], [1, 2], 'cpu');
      
      await expect(layer.backward(gradOutput)).rejects.toThrow();
    });
  });
  
  describe('weight updates', () => {
    it('should update weights correctly', async () => {
      const layer = new DenseLayer(2, 2, null, 'cpu');
      const originalWeights = layer.getWeights().data.slice();
      const originalBias = layer.getBias().data.slice();
      
      const gradWeights = await TensorOps.create(
        [0.1, 0.2, 0.3, 0.4], [2, 2], 'cpu'
      );
      const gradBias = await TensorOps.create([0.1, 0.2], [2], 'cpu');
      const learningRate = 0.01;
      
      await layer.updateWeights(gradWeights, gradBias, learningRate);
      
      const newWeights = layer.getWeights().data;
      const newBias = layer.getBias().data;
      
      // Check weights were updated
      for (let i = 0; i < originalWeights.length; i++) {
        expect(newWeights[i]).toBeCloseTo(
          originalWeights[i] - learningRate * gradWeights.data[i],
          5
        );
      }
      
      // Check bias was updated
      for (let i = 0; i < originalBias.length; i++) {
        expect(newBias[i]).toBeCloseTo(
          originalBias[i] - learningRate * gradBias.data[i],
          5
        );
      }
    });
  });
  
  describe('activation functions', () => {
    it('should work with sigmoid activation', async () => {
      const layer = new DenseLayer(2, 2, 'sigmoid', 'cpu');
      const input = await TensorOps.create([1, 2], [1, 2], 'cpu');
      
      const output = await layer.forward(input);
      
      // Sigmoid outputs should be between 0 and 1
      for (const value of output.data) {
        expect(value).toBeGreaterThan(0);
        expect(value).toBeLessThan(1);
      }
    });
    
    it('should work with tanh activation', async () => {
      const layer = new DenseLayer(2, 2, 'tanh', 'cpu');
      const input = await TensorOps.create([1, 2], [1, 2], 'cpu');
      
      const output = await layer.forward(input);
      
      // Tanh outputs should be between -1 and 1
      for (const value of output.data) {
        expect(value).toBeGreaterThan(-1);
        expect(value).toBeLessThan(1);
      }
    });
  });
});
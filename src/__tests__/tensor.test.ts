import { TensorOps } from '../core/tensor';
import { Tensor } from '../core/types';

describe('TensorOps', () => {
  describe('create', () => {
    it('should create CPU tensor with correct properties', async () => {
      const data = new Float32Array([1, 2, 3, 4]);
      const shape = [2, 2];
      
      const tensor = await TensorOps.create(data, shape, 'cpu');
      
      expect(tensor.data).toEqual(data);
      expect(tensor.shape).toEqual(shape);
      expect(tensor.device).toBe('cpu');
      expect(tensor.strides).toEqual([2, 1]);
    });
    
    it('should create tensor from number array', async () => {
      const data = [1, 2, 3, 4];
      const shape = [2, 2];
      
      const tensor = await TensorOps.create(data, shape, 'cpu');
      
      expect(tensor.data).toEqual(new Float32Array(data));
      expect(tensor.shape).toEqual(shape);
    });
  });
  
  describe('zeros', () => {
    it('should create zero-filled tensor', async () => {
      const shape = [2, 3];
      
      const tensor = await TensorOps.zeros(shape, 'cpu');
      
      expect(tensor.data).toEqual(new Float32Array([0, 0, 0, 0, 0, 0]));
      expect(tensor.shape).toEqual(shape);
    });
  });
  
  describe('ones', () => {
    it('should create one-filled tensor', async () => {
      const shape = [2, 2];
      
      const tensor = await TensorOps.ones(shape, 'cpu');
      
      expect(tensor.data).toEqual(new Float32Array([1, 1, 1, 1]));
      expect(tensor.shape).toEqual(shape);
    });
  });
  
  describe('random', () => {
    it('should create tensor with random values', async () => {
      const shape = [2, 3];
      
      const tensor = await TensorOps.random(shape, 'cpu');
      
      expect(tensor.shape).toEqual(shape);
      expect(tensor.data.length).toBe(6);
      
      // Check values are in reasonable range for Xavier initialization
      for (const value of tensor.data) {
        expect(Math.abs(value)).toBeLessThan(2);
      }
    });
  });
  
  describe('reshape', () => {
    it('should reshape tensor with same total size', async () => {
      const tensor = await TensorOps.create([1, 2, 3, 4, 5, 6], [2, 3], 'cpu');
      
      const reshaped = TensorOps.reshape(tensor, [3, 2]);
      
      expect(reshaped.shape).toEqual([3, 2]);
      expect(reshaped.data).toEqual(tensor.data);
      expect(reshaped.strides).toEqual([2, 1]);
    });
    
    it('should throw error for incompatible reshape', async () => {
      const tensor = await TensorOps.create([1, 2, 3, 4], [2, 2], 'cpu');
      
      expect(() => TensorOps.reshape(tensor, [2, 3])).toThrow();
    });
  });
  
  describe('slice', () => {
    it('should slice tensor correctly', async () => {
      const tensor = await TensorOps.create([1, 2, 3, 4, 5, 6], [2, 3], 'cpu');
      
      const sliced = TensorOps.slice(tensor, [0, 1], [2, 3]);
      
      expect(sliced.shape).toEqual([2, 2]);
      expect(sliced.data).toEqual(new Float32Array([2, 3, 5, 6]));
    });
    
    it('should handle single row slice', async () => {
      const tensor = await TensorOps.create([1, 2, 3, 4, 5, 6], [2, 3], 'cpu');
      
      const sliced = TensorOps.slice(tensor, [1, 0], [2, 3]);
      
      expect(sliced.shape).toEqual([1, 3]);
      expect(sliced.data).toEqual(new Float32Array([4, 5, 6]));
    });
  });
  
  describe('add', () => {
    it('should add two tensors element-wise', async () => {
      const a = await TensorOps.create([1, 2, 3, 4], [2, 2], 'cpu');
      const b = await TensorOps.create([2, 3, 4, 5], [2, 2], 'cpu');
      
      const result = await TensorOps.add(a, b);
      
      expect(result.data).toEqual(new Float32Array([3, 5, 7, 9]));
      expect(result.shape).toEqual([2, 2]);
    });
  });
  
  describe('multiply', () => {
    it('should multiply two tensors element-wise', async () => {
      const a = await TensorOps.create([1, 2, 3, 4], [2, 2], 'cpu');
      const b = await TensorOps.create([2, 3, 4, 5], [2, 2], 'cpu');
      
      const result = await TensorOps.multiply(a, b);
      
      expect(result.data).toEqual(new Float32Array([2, 6, 12, 20]));
      expect(result.shape).toEqual([2, 2]);
    });
  });
  
  describe('device conversion', () => {
    it('should convert CPU tensor to CPU (no-op)', async () => {
      const tensor = await TensorOps.create([1, 2, 3], [3], 'cpu');
      
      const result = await TensorOps.toCPU(tensor);
      
      expect(result).toBe(tensor);
    });
    
    it('should handle GPU tensor without WebGPU support gracefully', async () => {
      const tensor = await TensorOps.create([1, 2, 3], [3], 'cpu');
      
      // Should not throw even if WebGPU not available
      await expect(TensorOps.toGPU(tensor)).resolves.toBeDefined();
    });
  });
});
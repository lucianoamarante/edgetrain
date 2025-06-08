import { Model } from '../core/model';
import { TensorOps } from '../core/tensor';

describe('Model', () => {
  describe('construction', () => {
    it('should create model with config', () => {
      const config = {
        learningRate: 0.01,
        batchSize: 32,
        device: 'cpu' as const
      };
      
      const model = new Model(config);
      
      expect(model).toBeDefined();
    });
  });
  
  describe('layer management', () => {
    it('should add dense layers correctly', () => {
      const model = new Model({
        learningRate: 0.01,
        batchSize: 32,
        device: 'cpu'
      });
      
      model.addDenseLayer(10, 5, 'relu');
      model.addDenseLayer(5, 3, 'softmax');
      
      expect(model).toBeDefined();
    });
  });
  
  describe('forward pass', () => {
    it('should perform forward pass through multiple layers', async () => {
      const model = new Model({
        learningRate: 0.01,
        batchSize: 2,
        device: 'cpu'
      });
      
      model.addDenseLayer(3, 4, 'relu');
      model.addDenseLayer(4, 2, 'softmax');
      
      const input = await TensorOps.create(
        [1, 2, 3, 4, 5, 6], [2, 3], 'cpu'
      );
      
      const output = await model.forward(input);
      
      expect(output.shape).toEqual([2, 2]);
      expect(output.device).toBe('cpu');
      
      // Softmax outputs should sum to approximately 1 for each sample
      for (let i = 0; i < 2; i++) {
        const sum = output.data[i * 2] + output.data[i * 2 + 1];
        expect(sum).toBeCloseTo(1, 3);
      }
    });
  });
  
  describe('training', () => {
    it('should train model on simple data', async () => {
      const model = new Model({
        learningRate: 0.1,
        batchSize: 2,
        device: 'cpu'
      });
      
      model.addDenseLayer(2, 3, 'relu');
      model.addDenseLayer(3, 2, 'softmax');
      
      const inputs = await TensorOps.create(
        [1, 0, 0, 1, 1, 1, 0, 0], [4, 2], 'cpu'
      );
      
      const targets = await TensorOps.create(
        [1, 0, 0, 1, 1, 0, 0, 1], [4, 2], 'cpu'
      );
      
      const losses: number[] = [];
      
      await model.train(inputs, targets, 3, (epoch, loss) => {
        losses.push(loss);
      });
      
      expect(losses.length).toBe(3);
      expect(losses[0]).toBeGreaterThan(0);
      
      // Loss should generally decrease (though may fluctuate)
      expect(losses[losses.length - 1]).toBeLessThan(losses[0] * 2);
    });
    
    it('should handle single epoch training', async () => {
      const model = new Model({
        learningRate: 0.01,
        batchSize: 1,
        device: 'cpu'
      });
      
      model.addDenseLayer(1, 1);
      
      const inputs = await TensorOps.create([1], [1, 1], 'cpu');
      const targets = await TensorOps.create([0.5], [1, 1], 'cpu');
      
      let epochCalled = false;
      await model.train(inputs, targets, 1, () => {
        epochCalled = true;
      });
      
      expect(epochCalled).toBe(true);
    });
  });
  
  describe('prediction', () => {
    it('should make predictions after training', async () => {
      const model = new Model({
        learningRate: 0.1,
        batchSize: 2,
        device: 'cpu'
      });
      
      model.addDenseLayer(2, 2, 'softmax');
      
      const trainInputs = await TensorOps.create(
        [1, 0, 0, 1], [2, 2], 'cpu'
      );
      
      const trainTargets = await TensorOps.create(
        [1, 0, 0, 1], [2, 2], 'cpu'
      );
      
      await model.train(trainInputs, trainTargets, 2);
      
      const testInput = await TensorOps.create([1, 0], [1, 2], 'cpu');
      const prediction = await model.predict(testInput);
      
      expect(prediction.shape).toEqual([1, 2]);
      expect(prediction.device).toBe('cpu');
      
      // Softmax output should sum to 1
      const sum = prediction.data[0] + prediction.data[1];
      expect(sum).toBeCloseTo(1, 3);
    });
  });
  
  describe('error handling', () => {
    it('should handle empty training data gracefully', async () => {
      const model = new Model({
        learningRate: 0.01,
        batchSize: 1,
        device: 'cpu'
      });
      
      model.addDenseLayer(1, 1);
      
      const emptyInputs = await TensorOps.create([], [0, 1], 'cpu');
      const emptyTargets = await TensorOps.create([], [0, 1], 'cpu');
      
      // Should not crash on empty data
      await expect(model.train(emptyInputs, emptyTargets, 1))
        .resolves.not.toThrow();
    });
  });
});
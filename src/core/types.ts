export interface Tensor {
  data: Float32Array;
  shape: number[];
  strides: number[];
  device: 'cpu' | 'gpu';
  gpuBuffer?: GPUBuffer;
}

export interface Layer {
  forward(input: Tensor): Promise<Tensor>;
  backward(gradOutput: Tensor): Promise<Tensor>;
  getParameters(): Parameter[];
  setTraining(training: boolean): void;
}

export interface Parameter {
  data: Tensor;
  gradient?: Tensor;
  name: string;
  requires_grad: boolean;
}

export interface Optimizer {
  step(parameters: Parameter[]): Promise<void>;
  zero_grad(): void;
}

export interface TrainingConfig {
  epochs: number;
  batchSize: number;
  learningRate?: number;
  privacy?: PrivacyConfig;
  validation_split?: number;
  callbacks?: TrainingCallback[];
}

export interface PrivacyConfig {
  epsilon: number;
  delta: number;
  max_grad_norm?: number;
  noise_multiplier?: number;
}

export interface TrainingCallback {
  onEpochBegin?(epoch: number): void;
  onEpochEnd?(epoch: number, logs: TrainingLogs): void;
  onBatchBegin?(batch: number): void;
  onBatchEnd?(batch: number, logs: TrainingLogs): void;
}

export interface TrainingLogs {
  loss: number;
  accuracy?: number;
  val_loss?: number;
  val_accuracy?: number;
  privacy_budget?: number;
}

export interface GradientAggregator {
  addGradients(gradients: Parameter[]): void;
  getAggregatedGradients(): Promise<Parameter[]>;
  reset(): void;
}

export interface ModelArchitecture {
  layers: Layer[];
  optimizer: Optimizer;
  loss: LossFunction;
}

export interface LossFunction {
  forward(predictions: Tensor, targets: Tensor): Promise<number>;
  backward(predictions: Tensor, targets: Tensor): Promise<Tensor>;
}
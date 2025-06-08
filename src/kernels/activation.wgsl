@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> size: u32;

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= size) { return; }
  
  output[idx] = max(input[idx], 0.0);
}

@compute @workgroup_size(256)
fn relu_backward(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @group(0) @binding(3) var<storage, read> grad_output: array<f32>
) {
  let idx = global_id.x;
  if (idx >= size) { return; }
  
  if (input[idx] > 0.0) {
    output[idx] = grad_output[idx];
  } else {
    output[idx] = 0.0;
  }
}

@compute @workgroup_size(256)
fn sigmoid(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= size) { return; }
  
  output[idx] = 1.0 / (1.0 + exp(-input[idx]));
}

@compute @workgroup_size(256)
fn sigmoid_backward(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @group(0) @binding(3) var<storage, read> grad_output: array<f32>
) {
  let idx = global_id.x;
  if (idx >= size) { return; }
  
  let s = output[idx];
  output[idx] = grad_output[idx] * s * (1.0 - s);
}

@compute @workgroup_size(256)
fn tanh_activation(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= size) { return; }
  
  let exp_2x = exp(2.0 * input[idx]);
  output[idx] = (exp_2x - 1.0) / (exp_2x + 1.0);
}

@compute @workgroup_size(256)
fn softmax(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let batch_idx = global_id.x;
  let batch_size = size / 10u;
  
  if (batch_idx >= batch_size) { return; }
  
  let offset = batch_idx * 10u;
  
  var max_val = input[offset];
  for (var i = 1u; i < 10u; i = i + 1u) {
    max_val = max(max_val, input[offset + i]);
  }
  
  var sum = 0.0;
  for (var i = 0u; i < 10u; i = i + 1u) {
    let exp_val = exp(input[offset + i] - max_val);
    output[offset + i] = exp_val;
    sum = sum + exp_val;
  }
  
  for (var i = 0u; i < 10u; i = i + 1u) {
    output[offset + i] = output[offset + i] / sum;
  }
}
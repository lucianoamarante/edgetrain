// Mock WebGPU for testing environment
(globalThis as any).navigator = {
  ...(globalThis as any).navigator,
  gpu: undefined
};

// Global test setup
beforeEach(() => {
  // Reset any global state if needed
});

afterEach(() => {
  // Cleanup after each test
});
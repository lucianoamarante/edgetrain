module.exports = {
  parser: '@typescript-eslint/parser',
  extends: [
    'eslint:recommended'
  ],
  parserOptions: {
    ecmaVersion: 2022,
    sourceType: 'module'
  },
  env: {
    node: true,
    browser: true,
    es2022: true,
    jest: true
  },
  globals: {
    GPUComputePipeline: 'readonly',
    GPUShaderStage: 'readonly',
    GPUBufferBindingType: 'readonly',
    GPUBufferUsage: 'readonly',
    GPUDevice: 'readonly',
    GPUBuffer: 'readonly',
    GPUTexture: 'readonly',
    GPU: 'readonly'
  },
  rules: {
    // General code quality
    'no-console': 'warn',
    'no-debugger': 'error',
    'no-alert': 'error',
    'no-var': 'error',
    'prefer-const': 'error',
    'no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    
    // Error prevention
    'no-unreachable': 'error',
    'no-constant-condition': 'error',
    'no-unused-expressions': 'error',
    'no-throw-literal': 'error',
    
    // Code style
    'eqeqeq': ['error', 'always'],
    'curly': ['error', 'all'],
    'brace-style': ['error', '1tbs'],
    'comma-dangle': ['error', 'never'],
    'semi': ['error', 'always'],
    'quotes': ['error', 'single', { avoidEscape: true }],
    'indent': ['error', 2],
    'max-len': ['error', { code: 120, ignoreUrls: true }],
    
    // Security
    'no-eval': 'error',
    'no-implied-eval': 'error',
    'no-new-func': 'error',
    'no-script-url': 'error'
  },
  overrides: [
    {
      files: ['**/*.test.ts', '**/__tests__/**'],
      rules: {
        'max-len': 'off',
        'no-unused-expressions': 'off'
      }
    }
  ],
  ignorePatterns: [
    'dist/',
    'node_modules/',
    '*.wgsl',
    'demo/edgetrain.js'
  ]
};
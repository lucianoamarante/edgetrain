import typescript from '@rollup/plugin-typescript';
import resolve from '@rollup/plugin-node-resolve';

export default {
  input: 'src/index.ts',
  output: [
    {
      file: 'dist/index.js',
      format: 'es',
      sourcemap: true,
      inlineDynamicImports: true
    },
    {
      file: 'dist/index.umd.js',
      format: 'umd',
      name: 'EdgeTrain',
      sourcemap: true,
      inlineDynamicImports: true
    }
  ],
  plugins: [
    resolve(),
    typescript({
      tsconfig: './tsconfig.json'
    })
  ],
  external: []
};
# Contributing to EdgeTrain

Thank you for your interest in contributing to EdgeTrain. This document provides guidelines and information for contributors.

## Code of Conduct

This project follows a strict code of conduct. By participating, you agree to uphold this code.

## Development Setup

### Prerequisites

- Node.js 18.0 or higher
- npm 8.0 or higher
- Git

### Initial Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/v-code01/edgetrain.git
   cd edgetrain
   ```
3. Install dependencies:
   ```bash
   npm install
   ```
4. Install pre-commit hooks:
   ```bash
   npx pre-commit install
   ```

### Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run the full test suite:
   ```bash
   npm run test:coverage
   ```
4. Ensure code quality:
   ```bash
   npm run lint
   npm run typecheck
   npm run format:check
   ```
5. Build the project:
   ```bash
   npm run build
   ```
6. Commit your changes (pre-commit hooks will run automatically)
7. Push to your fork and create a pull request

## Code Standards

### TypeScript

- Use strict TypeScript configuration
- All functions must have explicit return types
- No `any` types allowed
- Prefer readonly types where applicable
- Use proper error handling with typed exceptions

### Code Style

- Follow ESLint configuration strictly
- Use Prettier for consistent formatting
- Maximum line length: 100 characters
- Maximum function length: 50 lines
- Maximum cyclomatic complexity: 10

### Testing

- Write comprehensive unit tests for all new features
- Maintain minimum 80% code coverage
- Test both CPU and GPU code paths
- Include edge cases and error conditions
- Use descriptive test names

### Documentation

- Document all public APIs with JSDoc
- Update README.md for new features
- Include code examples in documentation
- Keep documentation synchronized with code

## Pull Request Guidelines

### Before Submitting

- Ensure all tests pass
- Verify code coverage meets requirements
- Run security audit: `npm audit`
- Update documentation if needed
- Add changelog entry for significant changes

### PR Requirements

- Clear description of changes
- Reference any related issues
- Include tests for new functionality
- Maintain backward compatibility
- Follow semantic versioning guidelines

### Review Process

1. Automated checks must pass
2. Code review by maintainers
3. Testing on multiple platforms
4. Documentation review
5. Final approval and merge

## Types of Contributions

### Bug Reports

When filing bug reports, include:
- EdgeTrain version
- Browser and version
- WebGPU support status
- Minimal reproduction case
- Expected vs actual behavior
- System information

### Feature Requests

For feature requests, provide:
- Clear use case description
- Implementation suggestions
- Potential breaking changes
- Performance considerations
- Alternative solutions considered

### Code Contributions

Priority areas for contributions:
- WebGPU kernel optimizations
- Additional neural network layers
- Performance improvements
- Browser compatibility
- Documentation improvements

## Architecture Guidelines

### WebGPU Kernels

- Write efficient WGSL compute shaders
- Include both naive and optimized implementations
- Provide CPU fallbacks
- Test memory usage and performance
- Document shader parameters

### Tensor Operations

- Maintain shape and stride consistency
- Handle device transfers efficiently
- Implement proper error checking
- Support both sync and async operations
- Optimize for common use cases

### API Design

- Keep APIs simple and intuitive
- Maintain consistency across modules
- Provide comprehensive type definitions
- Support method chaining where appropriate
- Include helpful error messages

## Performance Considerations

- Profile GPU memory usage
- Minimize CPU-GPU transfers
- Use appropriate workgroup sizes
- Cache compiled shaders
- Optimize for common tensor shapes

## Security Guidelines

- Never commit credentials or tokens
- Validate all user inputs
- Use secure coding practices
- Audit dependencies regularly
- Report security issues privately

## Release Process

1. Update version in package.json
2. Update CHANGELOG.md
3. Create git tag
4. Publish to npm
5. Create GitHub release
6. Update documentation

## Getting Help

- GitHub Issues for bugs and features
- GitHub Discussions for questions
- Check existing documentation
- Review test cases for examples

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation
- Annual contributor summary

Thank you for contributing to EdgeTrain!
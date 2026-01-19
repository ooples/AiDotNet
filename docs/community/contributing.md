---
layout: default
title: Contributing
parent: Community
nav_order: 1
permalink: /community/contributing/
---

# Contributing to AiDotNet
{: .no_toc }

Thank you for your interest in contributing to AiDotNet! This guide will help you get started.
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Getting Started

### Prerequisites

- .NET 8.0 SDK or later
- Git
- A code editor (VS Code, Visual Studio, JetBrains Rider)
- Optional: CUDA Toolkit for GPU development

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/AiDotNet.git
cd AiDotNet
```

3. Add the upstream remote:

```bash
git remote add upstream https://github.com/ooples/AiDotNet.git
```

### Build the Project

```bash
dotnet build
```

### Run Tests

```bash
dotnet test
```

---

## Development Workflow

### Create a Branch

Always create a branch for your changes:

```bash
git checkout -b feature/your-feature-name
```

Use these prefixes:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation
- `refactor/` - Code refactoring
- `perf/` - Performance improvements
- `test/` - Test additions/changes

### Make Changes

1. Write clean, readable code
2. Follow the existing code style
3. Add XML documentation for public APIs
4. Include unit tests for new functionality

### Commit Guidelines

Write clear commit messages:

```
type(scope): Short description

Longer description if needed. Explain what and why,
not how.

Co-Authored-By: Your Name <your@email.com>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

### Submit a Pull Request

1. Push your branch:

```bash
git push origin feature/your-feature-name
```

2. Open a PR on GitHub
3. Fill out the PR template
4. Wait for review

---

## Code Style

### C# Guidelines

- Use modern C# features (pattern matching, records, etc.)
- Prefer expression-bodied members for simple methods
- Use nullable reference types
- Follow Microsoft naming conventions

```csharp
// Good
public class NeuralNetwork<T> where T : struct, INumber<T>
{
    private readonly ILayer<T>[] _layers;

    public int LayerCount => _layers.Length;

    public Tensor<T> Forward(Tensor<T> input)
    {
        var current = input;
        foreach (var layer in _layers)
        {
            current = layer.Forward(current);
        }
        return current;
    }
}
```

### Documentation

- Add XML documentation for all public types and members
- Include code examples in documentation
- Document exceptions that can be thrown

```csharp
/// <summary>
/// Applies the forward pass through all layers.
/// </summary>
/// <param name="input">The input tensor.</param>
/// <returns>The output tensor after passing through all layers.</returns>
/// <exception cref="ArgumentNullException">Thrown when input is null.</exception>
public Tensor<T> Forward(Tensor<T> input)
```

---

## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Mirror the source structure
- Use descriptive test names

```csharp
public class NeuralNetworkTests
{
    [Fact]
    public void Forward_WithValidInput_ReturnsExpectedShape()
    {
        // Arrange
        var network = new NeuralNetwork<float>(/* ... */);
        var input = Tensor<float>.Zeros(1, 10);

        // Act
        var output = network.Forward(input);

        // Assert
        Assert.Equal(new[] { 1, 5 }, output.Shape);
    }
}
```

### Running Specific Tests

```bash
# Run all tests
dotnet test

# Run specific test class
dotnet test --filter "NeuralNetworkTests"

# Run with verbose output
dotnet test --logger "console;verbosity=detailed"
```

---

## Areas to Contribute

### Good First Issues

Look for issues labeled `good first issue`:
- Documentation improvements
- Bug fixes with clear reproduction steps
- Test coverage improvements
- Code style fixes

### High-Impact Areas

- **Neural Network Architectures**: Add new model implementations
- **Computer Vision**: Object detection, segmentation models
- **Audio Processing**: Speech recognition, TTS improvements
- **Optimizers**: New optimization algorithms
- **Documentation**: Tutorials, examples, API docs

### Performance

- SIMD optimizations for tensor operations
- GPU kernel improvements
- Memory allocation optimizations
- Benchmark improvements

---

## PR Review Process

### What to Expect

1. **Automated Checks**: CI runs tests and linting
2. **Code Review**: Maintainers review your code
3. **Feedback**: You may receive suggestions or requests
4. **Approval**: Once approved, your PR will be merged

### Review Criteria

- Does it follow the code style?
- Are there adequate tests?
- Is the documentation updated?
- Does it maintain backward compatibility?
- Is the performance acceptable?

---

## Communication

### Where to Ask Questions

- **GitHub Discussions**: General questions and ideas
- **GitHub Issues**: Bug reports and feature requests
- **PR Comments**: Code-specific questions

### Be Respectful

- Be kind and constructive
- Assume good intentions
- Help others learn

---

## Recognition

Contributors are recognized in:
- Release notes for significant contributions
- The GitHub contributors page
- Special mentions for major features

Thank you for contributing to AiDotNet!

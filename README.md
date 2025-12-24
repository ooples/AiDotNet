# AiDotNet

<div align="center">

### Modern AI/ML Framework for .NET

**Bringing the latest AI algorithms and breakthroughs directly to the .NET ecosystem**

[![Build Status](https://github.com/ooples/AiDotNet/actions/workflows/ci.yml/badge.svg)](https://github.com/ooples/AiDotNet/actions/workflows/ci.yml)
[![CodeQL Analysis](https://github.com/ooples/AiDotNet/actions/workflows/codeql.yml/badge.svg)](https://github.com/ooples/AiDotNet/actions/workflows/codeql.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/59242b2fb53c4ffc871212d346de752f)](https://app.codacy.com/gh/ooples/AiDotNet/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![NuGet](https://img.shields.io/nuget/v/AiDotNet.svg)](https://www.nuget.org/packages/AiDotNet/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[Getting Started](#getting-started) •
[Documentation](#documentation) •
[Examples](#examples) •
[Contributing](#contributing)

</div>

---

## Overview

AiDotNet is a comprehensive machine learning and artificial intelligence library designed specifically for the .NET ecosystem. Our mission is to make cutting-edge AI algorithms accessible to .NET developers, whether you're a beginner taking your first steps in machine learning or an expert seeking full customization capabilities.

### Why AiDotNet?

- **Easy to Learn**: Simplified APIs that reduce the steep learning curve typically associated with AI/ML
- **Fully Customizable**: Expert users have complete control over algorithm parameters and implementation details
- **Modern Architecture**: Built with the latest .NET features and best practices
- **Production Ready**: Comprehensive testing, CI/CD pipelines, and quality gates ensure reliability
- **Actively Developed**: Regular updates bringing the latest AI breakthroughs to .NET

## Key Features

### 🧠 Neural Networks
- Flexible neural network architectures for classification and regression
- Support for custom layers and activation functions
- Advanced training with backpropagation and various optimizers

### 📈 Regression Models
- Linear and multiple regression
- Advanced regression techniques with feature engineering
- Real-world examples including housing price prediction

### ⏱️ Time Series Analysis
- Forecasting models for sequential data
- Support for stock prices, energy demand, and other time-dependent predictions
- Seasonal decomposition and trend analysis

### 🔄 Transfer Learning
- Domain adaptation algorithms
- Feature mapping between different data domains
- Pre-trained model support

### ⚡ Advanced Features
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning of large models
- **Automatic Differentiation**: Built-in autodiff for gradient computation
- **Distributed Training**: Scale your training across multiple machines
- **Mixed Precision Training**: Optimize performance with FP16/FP32 support
- **Language Models**: Integration with modern language model architectures
- **Agents**: AI agent frameworks for autonomous decision-making

### 🛠️ Supporting Components
- Multiple activation functions (ReLU, Sigmoid, Tanh, and more)
- Various optimization algorithms (Adam, SGD, RMSprop)
- Data preprocessing and normalization
- Outlier detection and removal
- Model evaluation metrics
- Caching for improved performance

## Getting Started

### Installation

Install AiDotNet via NuGet Package Manager:

```bash
dotnet add package AiDotNet
```

Or via the NuGet Package Manager Console:

```powershell
Install-Package AiDotNet
```

### Requirements

- .NET 8.0 or later
- .NET Framework 4.6.2 or later

### Quick Start

Here's a simple example to get you started with neural network classification:

```csharp
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;

// Create training data (XOR problem - classic neural network example)
var xorData = new double[,]
{
    { 0, 0 },  // Input: [0, 0]
    { 0, 1 },  // Input: [0, 1]
    { 1, 0 },  // Input: [1, 0]
    { 1, 1 }   // Input: [1, 1]
};

var xorLabels = new double[,]
{
    { 0 },  // Expected output: 0
    { 1 },  // Expected output: 1
    { 1 },  // Expected output: 1
    { 0 }   // Expected output: 0
};

// Convert to tensors (required format for neural network)
var features = new Tensor<double>(new int[] { 4, 2 }); // 4 samples, 2 features
var labels = new Tensor<double>(new int[] { 4, 1 });   // 4 samples, 1 output

for (int i = 0; i < 4; i++)
{
    for (int j = 0; j < 2; j++)
        features[new int[] { i, j }] = xorData[i, j];
    
    labels[new int[] { i, 0 }] = xorLabels[i, 0];
}

// Create neural network architecture
var architecture = new NeuralNetworkArchitecture<double>(
    inputFeatures: 2,
    numClasses: 1,
    complexity: NetworkComplexity.Medium
);

// Initialize and train the network
var neuralNetwork = new NeuralNetwork<double>(architecture);

for (int epoch = 0; epoch < 1000; epoch++)
{
    neuralNetwork.Train(features, labels);
    
    if (epoch % 200 == 0)
    {
        double loss = neuralNetwork.GetLastLoss();
        Console.WriteLine($"Epoch {epoch}: Loss = {loss:F4}");
    }
}

// Make predictions
var predictions = neuralNetwork.Predict(features);
Console.WriteLine($"
Prediction for [1, 0]: {predictions[new int[] { 2, 0 }]:F2}");
Console.WriteLine($"Prediction for [1, 1]: {predictions[new int[] { 3, 0 }]:F2}");
```

## Examples

AiDotNet comes with comprehensive examples demonstrating various use cases:

### Basic Examples
- **[Neural Network Example](testconsole/Examples/NeuralNetworkExample.cs)** - Classification (Iris dataset) and regression (housing prices)
- **[Regression Example](testconsole/Examples/RegressionExample.cs)** - Linear regression for house price prediction
- **[Time Series Example](testconsole/Examples/TimeSeriesExample.cs)** - Stock price forecasting

### Advanced Examples
- **[Enhanced Neural Network](testconsole/Examples/EnhancedNeuralNetworkExample.cs)** - Customer churn prediction with preprocessing
- **[Enhanced Regression](testconsole/Examples/EnhancedRegressionExample.cs)** - Real estate analysis with feature engineering
- **[Enhanced Time Series](testconsole/Examples/EnhancedTimeSeriesExample.cs)** - Energy demand forecasting with multiple models

### Specialized Examples
- **[Mixture of Experts](docs/examples/MixtureOfExpertsExample.md)** - Advanced ensemble learning techniques

To run the examples:

1. Clone the repository
2. Open `AiDotNet.sln` in Visual Studio or your preferred IDE
3. Set the `AiDotNetTestConsole` project as the startup project
4. Run the project and choose an example from the menu

## Documentation

- **[API Documentation](docs/)** - Comprehensive API reference
- **[Advanced Reasoning Guide](docs/AdvancedReasoningGuide.md)** - Deep dive into advanced ML concepts
- **[Distributed Training](docs/DistributedTrainingImplementations.md)** - Scale your training workloads
- **[Autodiff Integration](AUTODIFF_INTEGRATION.md)** - Understanding automatic differentiation
- **[Mixed Precision Architecture](MIXED_PRECISION_ARCHITECTURE.md)** - Optimize performance with mixed precision
- **[Physics-Informed Benchmarks](docs/PhysicsInformedBenchmarks.md)** - PDE and operator-learning benchmark harnesses

## Platform Support

| Platform | Versions |
|----------|----------|
| .NET | 8.0+ |
| .NET Framework | 4.6.2+ |
| Operating Systems | Windows, Linux, macOS |

## Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

Please read our [Contributing Guide](CONTRIBUTING.md) to learn about our development process and how to submit pull requests.

### Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Community & Support

- **Issues**: Found a bug or have a feature request? [Open an issue](https://github.com/ooples/AiDotNet/issues)
- **Discussions**: Have questions or want to discuss ideas? Start a [discussion](https://github.com/ooples/AiDotNet/discussions)
- **Security**: Found a security vulnerability? Please review our [Security Policy](SECURITY.md)

## Roadmap

AiDotNet is actively developed with regular updates. Current focus areas include:

- Expanding neural network architectures (CNNs, RNNs, Transformers)
- Additional optimization algorithms
- Enhanced GPU acceleration support
- More pre-built model templates
- Improved documentation and tutorials

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

AiDotNet is developed and maintained by [Ooples Finance](https://github.com/ooples) with contributions from the community. We're grateful to all our contributors who help make AI/ML more accessible in the .NET ecosystem.

---

<div align="center">

**Made with ❤️ for the .NET Community**

[⭐ Star us on GitHub](https://github.com/ooples/AiDotNet) • [📦 View on NuGet](https://www.nuget.org/packages/AiDotNet/)

</div>

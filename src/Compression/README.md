# Model Compression for AiDotNet

## Overview

The model compression module provides techniques to reduce model size and improve inference speed while maintaining reasonable accuracy. This is particularly important for deploying machine learning models in resource-constrained environments such as mobile devices, edge devices, or web browsers.

## Features

- **Multiple Compression Techniques:**
  - Quantization (8-bit, 4-bit, and mixed precision)
  - Pruning (removing unimportant connections)
  - Knowledge Distillation (training smaller models to mimic larger ones)
  - Low Rank Factorization (decomposing weight matrices)
  - Weight Clustering (grouping similar weights)
  - Tensor Decomposition (for compressing convolutional layers)
  - Huffman Coding (entropy-based encoding of parameters)

- **Evaluation Framework:**
  - Comprehensive comparison of model size before and after compression
  - Accuracy impact measurement
  - Inference speed benchmarking
  - Detailed metrics for each compression technique

- **Serialization Support:**
  - Efficient storage formats for compressed models
  - Preservation of compression benefits in serialized form
  - Fast loading of compressed models for inference

- **Logging and Benchmarking:**
  - Detailed logging of compression processes
  - Progress tracking and metrics collection during compression
  - Benchmarking tools for comparing different compression approaches
  - Automated benchmark reporting to aid decision-making

## Getting Started

### Basic Usage Example

```csharp
// Create a model compressor for the specific compression technique
var compressor = new QuantizationCompressor<NeuralNetworkModel<float>, Tensor<float>, Tensor<float>>();

// Configure compression options
var options = new ModelCompressionOptions {
    CompressionTechnique = CompressionTechnique.Quantization,
    QuantizationPrecision = 8,
    MaxAcceptableAccuracyLoss = 0.02,
    LoggingOptions = new LoggingOptions { 
        IsEnabled = true,
        MinimumLevel = LoggingLevel.Information,
        LogToConsole = true
    }
};

// Apply compression
var compressedModel = compressor.Compress(originalModel, options);

// Evaluate compression results
var results = compressor.EvaluateCompression(
    originalModel, compressedModel, testInputs, testOutputs);

// Print results
Console.WriteLine(results.ToString());

// Save the compressed model
compressor.SerializeCompressedModel(compressedModel, "compressed_model.bin");

// Later, load the compressed model
var loadedModel = compressor.DeserializeCompressedModel("compressed_model.bin");
```

### Choosing the Right Technique

| Technique | Size Reduction | Accuracy Impact | Speed Improvement | Best For |
|-----------|----------------|-----------------|-------------------|----------|
| Quantization (8-bit) | 3-4x | Low (< 1%) | 1.5-2.5x | General use, hardware with int8 support |
| Pruning (70% sparsity) | 2-3x | Low-Moderate (1-3%) | 1.2-1.8x | Overparameterized models |
| Knowledge Distillation | 3-10x | Moderate (2-5%) | 2-5x | Very large models, when retraining is possible |
| Low Rank Factorization | 2-4x | Moderate (1-3%) | 1.2-2.0x | Models with large fully-connected layers |
| Quantized Pruning | 5-8x | Moderate (2-4%) | 2-3x | Maximum compression needs |

## Implementing for Custom Model Types

To enable compression for custom model types:

1. **For Quantization:**
   - Implement `IQuantizableModel<TModel>` on your model class
   - Register a factory with `QuantizedModelFactoryRegistry`

2. **For Other Techniques:**
   - Implement technique-specific interfaces for your model
   - Extend the relevant compressor base class

## Production Best Practices

1. **Always Verify Accuracy:**
   - Set `ValidateAfterCompression = true` in compression options
   - Monitor `AccuracyImpact` in the compression results
   - Fine-tune if accuracy loss exceeds acceptable threshold

2. **Use Mixed Precision When Possible:**
   - Set `UseMixedPrecision = true` for better accuracy-size trade-off
   - Critical model components will use higher precision
   - Less sensitive parts will use lower precision

3. **Combine Techniques:**
   - Apply quantization after pruning for maximum benefits
   - Consider knowledge distillation for very large models

4. **Customize for Target Hardware:**
   - Use 8-bit quantization for hardware with int8 acceleration
   - For extremely constrained devices, consider 4-bit quantization
   - Enable sparse execution optimizations for pruned models

5. **Enable Logging for Production Environments:**
   - Configure appropriate log levels based on environment
   - Use `Information` level for production monitoring
   - Use `Debug` level for development and troubleshooting
   - Archive logs for post-mortem analysis if issues occur

6. **Benchmark Different Approaches:**
   - Use `CompressionBenchmarkLogger` to record results
   - Compare multiple techniques to find the best trade-off
   - Generate reports to document your evaluation process

## Logging and Benchmarking

### Compression Logging

The compression framework includes a robust logging system that tracks:

- Original and compressed model sizes
- Compression ratio achieved
- Accuracy impact measurements
- Inference speed improvements
- Hardware-specific optimizations
- Detailed progress during compression operations

Example:

```csharp
// Configure logging options
var loggingOptions = new LoggingOptions
{
    IsEnabled = true,
    MinimumLevel = LoggingLevel.Debug,
    LogDirectory = "Logs",
    LogToConsole = true
};

// Create compression options with logging enabled
var options = new ModelCompressionOptions
{
    CompressionTechnique = CompressionTechnique.Quantization,
    LoggingOptions = loggingOptions
};

// Apply compression with logging
var compressedModel = compressor.Compress(originalModel, options);
```

### Compression Benchmarking

The benchmarking system allows you to:

- Record and compare results from different compression techniques
- Generate detailed reports on compression performance
- Identify optimal compression approaches for specific models
- Save and load benchmark results for future reference

Example:

```csharp
// Create a benchmark logger
var benchmarkLogger = new CompressionBenchmarkLogger(loggingOptions);

// Record benchmark results
benchmarkLogger.RecordBenchmarkResult(new CompressionBenchmarkResult
{
    ModelName = "MobileNetV2",
    CompressionTechnique = "Quantization",
    OriginalSizeBytes = 14_000_000,
    CompressedSizeBytes = 3_500_000,
    CompressionRatio = 4.0,
    AccuracyImpact = -0.015,
    InferenceSpeedup = 1.8,
    MemoryReduction = 0.75
});

// Generate a benchmark report
var reportPath = benchmarkLogger.GenerateBenchmarkReport();
```

## Advanced Topics

- **Dynamic Quantization:**
  Set `IsDynamicCompression = true` to apply quantization during inference rather than ahead of time

- **Calibration:**
  Use representative data for calibration to optimize dynamic range for quantization

- **Fine-tuning Compressed Models:**
  Perform additional training on compressed models to recover accuracy

- **Custom Serialization:**
  Implement custom serialization for specific deployment targets

- **Hardware-Specific Optimization:**
  Configure `HardwareOptimization` options to leverage platform-specific capabilities:
  ```csharp
  var options = new ModelCompressionOptions {
      HardwareOptimization = new HardwareOptimizationOptions {
          EnableSIMD = true,
          EnableAVX = true,
          TargetGPU = false,
          OptimizeForLowPower = true
      }
  };
  ```

## Contributing

We welcome contributions to expand the range of compression techniques and improve their effectiveness. See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.
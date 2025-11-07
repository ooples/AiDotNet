# AiDotNet Deployment Guide

This guide covers deploying AiDotNet models to production environments including TensorRT, mobile devices (iOS/Android), and edge devices.

## Table of Contents

- [Overview](#overview)
- [ONNX Export](#onnx-export)
- [TensorRT Deployment](#tensorrt-deployment)
- [Mobile Deployment](#mobile-deployment)
  - [iOS with CoreML](#ios-with-coreml)
  - [Android with TensorFlow Lite](#android-with-tensorflow-lite)
  - [Android with NNAPI](#android-with-nnapi)
- [Edge Device Optimization](#edge-device-optimization)
- [Runtime Features](#runtime-features)
- [Performance Optimization](#performance-optimization)

## Overview

AiDotNet provides comprehensive deployment capabilities for various platforms:

- **TensorRT**: High-performance GPU inference with 5-10x speedup
- **CoreML**: Native iOS deployment with Neural Engine support
- **TensorFlow Lite**: Cross-platform mobile deployment
- **NNAPI**: Hardware-accelerated Android inference
- **Edge Optimization**: ARM NEON, model partitioning, adaptive inference

## ONNX Export

ONNX is the universal format that serves as the foundation for all other export formats.

### Basic Export

```csharp
using AiDotNet.Deployment.Export;
using AiDotNet.Deployment.Export.Onnx;

// Create your model
var model = new NeuralNetworkModel<float>();
// ... train your model ...

// Export to ONNX
var exporter = new OnnxModelExporter<float>();
var config = new ExportConfiguration
{
    ModelName = "MyModel",
    ModelVersion = "1.0",
    OptimizeModel = true,
    InputShape = new[] { 224, 224, 3 }
};

exporter.Export(model, "model.onnx", config);
```

### Dynamic Shapes

```csharp
var config = new ExportConfiguration
{
    UseDynamicShapes = true,
    InputShape = new[] { 224, 224, 3 }
};
exporter.Export(model, "model_dynamic.onnx", config);
```

## TensorRT Deployment

TensorRT provides optimized GPU inference for NVIDIA GPUs with 5-10x performance improvements.

### Convert to TensorRT

```csharp
using AiDotNet.Deployment.TensorRT;

var converter = new TensorRTConverter<float>();
var config = TensorRTConfiguration.ForMaxPerformance();

// Convert model to TensorRT engine
converter.ConvertToTensorRT(model, "model.engine", config);
```

### Inference with TensorRT

```csharp
var engine = new TensorRTInferenceEngine<float>("model.engine", config);
engine.Initialize();

// Warm up
engine.WarmUp(numIterations: 10);

// Run inference
var input = new float[224 * 224 * 3];
var output = await engine.InferAsync(input);

// Batch inference
var inputs = new float[32][];
var outputs = await engine.InferBatchAsync(inputs);
```

### TensorRT with INT8 Quantization

```csharp
var config = TensorRTConfiguration.ForInt8("calibration_data.bin");
config.MaxBatchSize = 8;
config.MaxWorkspaceSize = 2L << 30; // 2 GB

converter.ConvertToTensorRT(model, "model_int8.engine", config);
```

### Multi-Stream Execution

```csharp
var config = new TensorRTConfiguration
{
    EnableMultiStream = true,
    NumStreams = 4,
    EnableCudaGraphs = true,
    MaxBatchSize = 32
};

var engine = new TensorRTInferenceEngine<float>("model.engine", config);
// Automatically handles concurrent inference across multiple streams
```

## Mobile Deployment

### iOS with CoreML

#### Export to CoreML

```csharp
using AiDotNet.Deployment.Mobile.CoreML;

var exporter = new CoreMLExporter<float>();
var config = CoreMLConfiguration.ForIPhone();

exporter.ExportToCoreML(model, "model.mlmodel", config);
```

#### Neural Engine Optimization

```csharp
var config = CoreMLConfiguration.ForNeuralEngine();
config.UseQuantization = true;
config.QuantizationBits = 8;
config.OptimizeForSize = true;

exporter.ExportToCoreML(model, "model_optimized.mlmodel", config);
```

#### iPad Configuration

```csharp
var config = CoreMLConfiguration.ForIPad();
config.QuantizationBits = 16; // Higher quality on iPad
config.OptimizeForSize = false;
```

### Android with TensorFlow Lite

#### Export to TFLite

```csharp
using AiDotNet.Deployment.Mobile.TensorFlowLite;

var exporter = new TFLiteExporter<float>();
var config = TFLiteConfiguration.ForAndroid();

exporter.ExportToTFLite(model, "model.tflite", config);
```

#### Integer-Only Quantization

```csharp
var config = TFLiteConfiguration.ForIntegerOnly();
config.UseIntegerOnlyQuantization = true;
config.EnableOperatorFusion = true;

exporter.ExportToTFLite(model, "model_int8.tflite", config);
```

#### GPU Delegate

```csharp
var config = new TFLiteConfiguration
{
    UseGpuDelegate = true,
    UseXnnpackDelegate = true,
    NumThreads = 4
};
```

### Android with NNAPI

NNAPI provides hardware acceleration on Android devices (GPU, DSP, NPU).

```csharp
using AiDotNet.Deployment.Mobile.Android;

var backend = new NNAPIBackend<float>(NNAPIConfiguration.ForMaxPerformance());
backend.Initialize();
backend.LoadModel("model.tflite");

var input = new float[224 * 224 * 3];
var output = await backend.ExecuteAsync(input);
```

#### Low Power Configuration

```csharp
var config = NNAPIConfiguration.ForLowPower();
config.PreferredDevice = NNAPIDevice.DSP;
config.ExecutionPreference = NNAPIExecutionPreference.LowPower;

var backend = new NNAPIBackend<float>(config);
```

## Edge Device Optimization

### ARM NEON Optimization

```csharp
using AiDotNet.Deployment.Edge;

var config = EdgeConfiguration.ForRaspberryPi();
config.EnableArmNeonOptimization = true;

var optimizer = new EdgeOptimizer<float>(config);
var optimizedModel = optimizer.OptimizeForEdge(model);
```

### Model Partitioning (Cloud + Edge)

```csharp
var config = EdgeConfiguration.ForCloudEdge();
config.EnableModelPartitioning = true;
config.PartitionStrategy = PartitionStrategy.Adaptive;

var optimizer = new EdgeOptimizer<float>(config);
var partitioned = optimizer.PartitionModel(model);

// Deploy edge part to device
// Deploy cloud part to server
var edgeModel = partitioned.EdgeModel;
var cloudModel = partitioned.CloudModel;
```

### Adaptive Inference

```csharp
var optimizer = new EdgeOptimizer<float>(config);

// Adjust quality based on battery and CPU
var batteryLevel = 0.3; // 30%
var cpuLoad = 0.8; // 80%

var adaptiveConfig = optimizer.CreateAdaptiveConfig(batteryLevel, cpuLoad);
// Automatically adjusts quality/speed tradeoff
```

### Device-Specific Configurations

#### Raspberry Pi

```csharp
var config = EdgeConfiguration.ForRaspberryPi();
// Optimized for ARM Cortex-A, uses INT8, ARM NEON
```

#### NVIDIA Jetson

```csharp
var config = EdgeConfiguration.ForJetson();
// Optimized for Jetson GPU, uses FP16
```

#### Microcontroller

```csharp
var config = EdgeConfiguration.ForMicrocontroller();
// Maximum compression, 1MB model size limit
```

## Runtime Features

### Model Versioning

```csharp
using AiDotNet.Deployment.Runtime;

var runtime = new DeploymentRuntime<float>(RuntimeConfiguration.ForProduction());

// Register multiple versions
runtime.RegisterModel("MyModel", "1.0", "model_v1.onnx");
runtime.RegisterModel("MyModel", "1.1", "model_v1.1.onnx");
runtime.RegisterModel("MyModel", "2.0", "model_v2.onnx");

// Use specific version
var output = await runtime.InferAsync("MyModel", "1.1", input);

// Use latest version
var output = await runtime.InferAsync("MyModel", "latest", input);
```

### Model Warm-Up

```csharp
// Manual warm-up
runtime.WarmUpModel("MyModel", "1.0", numIterations: 20);

// Or enable auto warm-up
var config = new RuntimeConfiguration
{
    AutoWarmUp = true,
    WarmUpIterations = 10
};
```

### A/B Testing

```csharp
// Set up A/B test
runtime.SetupABTest(
    testName: "ModelComparison",
    modelName: "MyModel",
    versionA: "1.0",
    versionB: "2.0",
    trafficSplit: 0.5 // 50/50 split
);

// Run inference with A/B test
var (output, selectedVersion) = await runtime.InferWithABTestAsync("ModelComparison", input);

// Get statistics
var stats = runtime.GetModelStatistics("MyModel");
Console.WriteLine($"Average latency: {stats.AverageLatencyMs}ms");
Console.WriteLine($"Error rate: {stats.ErrorRate * 100}%");
```

### Telemetry and Monitoring

```csharp
var config = new RuntimeConfiguration
{
    EnableTelemetry = true,
    TelemetrySamplingRate = 0.1, // Sample 10%
    EnablePerformanceMonitoring = true,
    PerformanceAlertThresholdMs = 500.0
};

var runtime = new DeploymentRuntime<float>(config);

// Get statistics
var stats = runtime.GetModelStatistics("MyModel", "1.0");
Console.WriteLine($"Total inferences: {stats.TotalInferences}");
Console.WriteLine($"Cache hit rate: {stats.CacheHitRate * 100}%");
Console.WriteLine($"Min/Avg/Max latency: {stats.MinLatencyMs}/{stats.AverageLatencyMs}/{stats.MaxLatencyMs}ms");
```

## Performance Optimization

### Quantization

```csharp
using AiDotNet.Deployment.Optimization.Quantization;

// INT8 Quantization
var quantizer = new Int8Quantizer<float>();

// Calibrate with sample data
var calibrationData = GetCalibrationSamples();
quantizer.Calibrate(calibrationData);

// Quantize model
var quantConfig = QuantizationConfiguration.ForInt8(CalibrationMethod.Entropy);
var quantizedModel = quantizer.Quantize(model, quantConfig);

// Float16 Quantization (no calibration needed)
var fp16Quantizer = new Float16Quantizer<float>();
var fp16Config = QuantizationConfiguration.ForFloat16();
var fp16Model = fp16Quantizer.Quantize(model, fp16Config);
```

### Complete Deployment Pipeline Example

```csharp
using AiDotNet.Deployment.Export;
using AiDotNet.Deployment.TensorRT;
using AiDotNet.Deployment.Optimization.Quantization;
using AiDotNet.Deployment.Runtime;

public class DeploymentPipeline
{
    public async Task DeployModel(NeuralNetworkModel<float> model, string deploymentPath)
    {
        // 1. Quantize the model
        var quantizer = new Int8Quantizer<float>();
        var calibrationData = LoadCalibrationData();
        quantizer.Calibrate(calibrationData);

        var quantConfig = QuantizationConfiguration.ForInt8(CalibrationMethod.Entropy);
        var quantizedModel = quantizer.Quantize(model, quantConfig);

        // 2. Export to ONNX
        var onnxExporter = new OnnxModelExporter<float>();
        var exportConfig = ExportConfiguration.ForTensorRT(batchSize: 8, useFp16: true);
        onnxExporter.Export(quantizedModel, $"{deploymentPath}/model.onnx", exportConfig);

        // 3. Convert to TensorRT
        var trtConverter = new TensorRTConverter<float>();
        var trtConfig = TensorRTConfiguration.ForMaxPerformance();
        trtConverter.ConvertToTensorRT(quantizedModel, $"{deploymentPath}/model.engine", trtConfig);

        // 4. Set up runtime
        var runtime = new DeploymentRuntime<float>(RuntimeConfiguration.ForProduction());
        runtime.RegisterModel("MyModel", "1.0", $"{deploymentPath}/model.engine");

        // 5. Warm up
        runtime.WarmUpModel("MyModel", "1.0", numIterations: 20);

        // 6. Ready for inference
        Console.WriteLine("Model deployed and ready!");

        // Get statistics
        var stats = runtime.GetModelStatistics("MyModel", "1.0");
        Console.WriteLine($"Warm-up time: {stats.MinLatencyMs}ms");
    }
}
```

## Best Practices

1. **Always quantize for mobile/edge**: Use INT8 or FP16 for significant size and speed improvements
2. **Warm up models**: Run warm-up iterations before production to eliminate cold-start latency
3. **Use caching**: Enable inference caching for repeated inputs
4. **Monitor performance**: Track latency, error rates, and cache hit rates
5. **Test on target hardware**: Performance varies significantly across devices
6. **Version your models**: Use semantic versioning and maintain multiple versions
7. **Implement A/B testing**: Compare model versions in production before full rollout
8. **Optimize batch sizes**: Find optimal batch size for your workload and hardware

## Troubleshooting

### Model Export Fails

- Check that all layers are supported for the target format
- Verify input shapes are correctly specified
- Ensure the model is properly trained and saved

### Poor Performance

- Enable quantization (INT8 or FP16)
- Increase batch size if processing multiple requests
- Use hardware acceleration (TensorRT, NNAPI, CoreML)
- Check that warm-up has been performed

### High Memory Usage

- Reduce model size with quantization and pruning
- Decrease cache size in RuntimeConfiguration
- Use model partitioning for edge devices
- Disable caching if memory is constrained

## Next Steps

- Review the [Performance Benchmarks](./BENCHMARKS.md) (coming soon)
- Check out [Advanced Examples](./Examples/) (coming soon)
- Read about [Custom Operator Support](./CUSTOM_OPERATORS.md) (coming soon)

## Support

For issues or questions:
- GitHub Issues: https://github.com/ooples/AiDotNet/issues
- Documentation: https://github.com/ooples/AiDotNet/wiki

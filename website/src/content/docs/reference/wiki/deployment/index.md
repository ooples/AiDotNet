---
title: "Deployment"
description: "All 73 public types in the AiDotNet.deployment namespace, organized by kind."
section: "API Reference"
---

**73** public types in this namespace, organized by kind.

## Models & Types (41)

| Type | Summary |
|:-----|:--------|
| [`ABTest`](/docs/reference/wiki/deployment/abtest/) | Represents a single A/B test configuration. |
| [`AWQQuantizer<T, TInput, TOutput>`](/docs/reference/wiki/deployment/awqquantizer/) | AWQ (Activation-aware Weight Quantization) - protects important weights based on activation magnitudes. |
| [`ActivationStatistics<T>`](/docs/reference/wiki/deployment/activationstatistics/) | Holds activation statistics collected during calibration forward passes. |
| [`BlockQuantizationState`](/docs/reference/wiki/deployment/blockquantizationstate/) | Stores block-wise quantization state for efficient QAT. |
| [`CacheStatistics`](/docs/reference/wiki/deployment/cachestatistics/) | Statistics for the model cache. |
| [`CoreMLExporter<T, TInput, TOutput>`](/docs/reference/wiki/deployment/coremlexporter/) | Exports models to CoreML format for iOS deployment. |
| [`DeploymentRuntime<T>`](/docs/reference/wiki/deployment/deploymentruntime/) | Runtime environment for deployed models with warm-up, versioning, A/B testing, and telemetry. |
| [`EdgeOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/deployment/edgeoptimizer/) | Optimizer for edge device deployment with ARM NEON and other optimizations. |
| [`EfficientQATOptimizer<T>`](/docs/reference/wiki/deployment/efficientqatoptimizer/) | EfficientQAT optimizer providing memory-efficient Quantization-Aware Training for large models. |
| [`FP8Quantizer<T, TInput, TOutput>`](/docs/reference/wiki/deployment/fp8quantizer/) | FP8 (8-bit Floating Point) quantizer supporting E4M3 and E5M2 formats. |
| [`Float16Quantizer<T, TInput, TOutput>`](/docs/reference/wiki/deployment/float16quantizer/) | FP16 (half-precision) quantizer for model optimization. |
| [`GPTQQuantizer<T, TInput, TOutput>`](/docs/reference/wiki/deployment/gptqquantizer/) | GPTQ (Generative Pre-trained Transformer Quantization) - state-of-the-art weight quantization using second-order Hessian information to minimize reconstruction error. |
| [`InferenceStatistics`](/docs/reference/wiki/deployment/inferencestatistics/) | Statistics for TensorRT inference engine. |
| [`Int8Quantizer<T, TInput, TOutput>`](/docs/reference/wiki/deployment/int8quantizer/) | INT8 quantizer for model optimization. |
| [`LayerActivationStats<T>`](/docs/reference/wiki/deployment/layeractivationstats/) | Activation statistics for a single layer. |
| [`LayerQuantizationParams`](/docs/reference/wiki/deployment/layerquantizationparams/) | Per-layer quantization parameters. |
| [`MXFP4Quantizer<T, TInput, TOutput>`](/docs/reference/wiki/deployment/mxfp4quantizer/) | MXFP4 (Microscaling FP4) quantizer - uses shared exponents for efficient 4-bit floating point. |
| [`ModelCache<T>`](/docs/reference/wiki/deployment/modelcache/) | Cache for model inference results. |
| [`ModelStatistics`](/docs/reference/wiki/deployment/modelstatistics/) | Statistics for a model. |
| [`ModelVersionInfo`](/docs/reference/wiki/deployment/modelversioninfo/) | Public model version information. |
| [`NF4Quantizer<T, TInput, TOutput>`](/docs/reference/wiki/deployment/nf4quantizer/) | NF4 (4-bit NormalFloat) quantizer - optimal for normally distributed weights. |
| [`NNAPIBackend<T>`](/docs/reference/wiki/deployment/nnapibackend/) | NNAPI (Neural Networks API) backend for Android deployment. |
| [`NNAPIDeviceInfo`](/docs/reference/wiki/deployment/nnapideviceinfo/) | Information about an NNAPI-capable device. |
| [`NNAPIPerformanceInfo`](/docs/reference/wiki/deployment/nnapiperformanceinfo/) | Performance information for NNAPI. |
| [`OnnxGraph`](/docs/reference/wiki/deployment/onnxgraph/) | Represents an ONNX computational graph. |
| [`OnnxModelExporter<T, TInput, TOutput>`](/docs/reference/wiki/deployment/onnxmodelexporter/) | Exports AiDotNet models to ONNX format for cross-platform deployment. |
| [`OnnxOperation`](/docs/reference/wiki/deployment/onnxoperation/) | Represents an ONNX operation (node in the computational graph). |
| [`OptimizationProfile`](/docs/reference/wiki/deployment/optimizationprofile/) | Represents a TensorRT optimization profile for dynamic shapes. |
| [`PartitionedModel<T, TInput, TOutput>`](/docs/reference/wiki/deployment/partitionedmodel/) | Represents a model partitioned for cloud+edge deployment. |
| [`QATTrainingHook<T>`](/docs/reference/wiki/deployment/qattraininghook/) | Quantization-Aware Training (QAT) hook that applies fake quantization during training. |
| [`QuIPSharpQuantizer<T, TInput, TOutput>`](/docs/reference/wiki/deployment/quipsharpquantizer/) | QuIP# (Quantization with Incoherence Processing Sharp) quantizer for extreme 2-bit quantization. |
| [`QuantizationState`](/docs/reference/wiki/deployment/quantizationstate/) | Stores quantization state for a layer during QAT. |
| [`SmoothQuantQuantizer<T, TInput, TOutput>`](/docs/reference/wiki/deployment/smoothquantquantizer/) | SmoothQuant - transfers quantization difficulty from activations to weights using per-channel smoothing. |
| [`SpinQuantQuantizer<T, TInput, TOutput>`](/docs/reference/wiki/deployment/spinquantquantizer/) | SpinQuant quantizer - uses learned rotation matrices to reduce outliers before quantization. |
| [`TFLiteExporter<T, TInput, TOutput>`](/docs/reference/wiki/deployment/tfliteexporter/) | Exports models to TensorFlow Lite format for mobile deployment. |
| [`TFLiteTargetSpec`](/docs/reference/wiki/deployment/tflitetargetspec/) | TensorFlow Lite target specification with detailed platform requirements. |
| [`TelemetryCollector`](/docs/reference/wiki/deployment/telemetrycollector/) | Collects telemetry data for deployed models. |
| [`TelemetryEvent`](/docs/reference/wiki/deployment/telemetryevent/) | Represents a telemetry event. |
| [`TensorRTConverter<T, TInput, TOutput>`](/docs/reference/wiki/deployment/tensorrtconverter/) | Converts models to TensorRT optimized format for NVIDIA GPU deployment. |
| [`TensorRTInferenceEngine<T>`](/docs/reference/wiki/deployment/tensorrtinferenceengine/) | High-performance inference engine for TensorRT models. |
| [`WeightStreamingReport`](/docs/reference/wiki/deployment/weightstreamingreport/) | Telemetry summary for a model's weight-streaming activity. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`ModelExporterBase<T, TInput, TOutput>`](/docs/reference/wiki/deployment/modelexporterbase/) | Abstract base class for model exporters that provides common functionality. |

## Interfaces (3)

| Type | Summary |
|:-----|:--------|
| [`IModelExporter<T, TInput, TOutput>`](/docs/reference/wiki/deployment/imodelexporter/) | Base interface for model exporters that convert AiDotNet models to various deployment formats. |
| [`INNAPIGraphBuilder`](/docs/reference/wiki/deployment/innapigraphbuilder/) | Translates a model file (TFLite / ONNX / etc.) into an NNAPI operation graph by adding operands and operations to a freshly-created `ANeuralNetworksModel` handle. |
| [`IQuantizer<T, TInput, TOutput>`](/docs/reference/wiki/deployment/iquantizer/) | Interface for model quantization strategies. |

## Enums (6)

| Type | Summary |
|:-----|:--------|
| [`CoreMLComputeUnits`](/docs/reference/wiki/deployment/coremlcomputeunits/) | CoreML compute unit options. |
| [`FP8Format`](/docs/reference/wiki/deployment/fp8format/) | Specifies the FP8 format variant to use. |
| [`NNAPIDevice`](/docs/reference/wiki/deployment/nnapidevice/) | NNAPI acceleration devices. |
| [`NNAPIExecutionPreference`](/docs/reference/wiki/deployment/nnapiexecutionpreference/) | NNAPI execution preferences. |
| [`TFLiteTargetType`](/docs/reference/wiki/deployment/tflitetargettype/) | TensorFlow Lite target type for compatibility. |
| [`TensorRTPrecision`](/docs/reference/wiki/deployment/tensorrtprecision/) | TensorRT precision modes for inference. |

## Options & Configuration (20)

| Type | Summary |
|:-----|:--------|
| [`ABTestingConfig`](/docs/reference/wiki/deployment/abtestingconfig/) | Configuration for A/B testing - comparing multiple model versions by splitting traffic. |
| [`AdaptiveInferenceConfig`](/docs/reference/wiki/deployment/adaptiveinferenceconfig/) | Configuration for adaptive inference. |
| [`CacheConfig`](/docs/reference/wiki/deployment/cacheconfig/) | Configuration for model caching - storing loaded models in memory to avoid repeated loading. |
| [`CompressionConfig`](/docs/reference/wiki/deployment/compressionconfig/) | Configuration for model compression - reducing model size while preserving accuracy. |
| [`CoreMLConfiguration`](/docs/reference/wiki/deployment/coremlconfiguration/) | Configuration for CoreML model export. |
| [`DeploymentConfiguration`](/docs/reference/wiki/deployment/deploymentconfiguration/) | Aggregates all deployment-related configurations. |
| [`EdgeConfiguration`](/docs/reference/wiki/deployment/edgeconfiguration/) | Configuration for edge device deployment optimization. |
| [`ExportConfig`](/docs/reference/wiki/deployment/exportconfig/) | Configuration for exporting models to different formats and platforms. |
| [`ExportConfiguration`](/docs/reference/wiki/deployment/exportconfiguration/) | Configuration options for model export operations. |
| [`NNAPIConfiguration`](/docs/reference/wiki/deployment/nnapiconfiguration/) | Configuration for NNAPI backend. |
| [`OptimizationProfileConfig`](/docs/reference/wiki/deployment/optimizationprofileconfig/) | Configuration for a single optimization profile (for dynamic shapes). |
| [`ProfilingConfig`](/docs/reference/wiki/deployment/profilingconfig/) | Configuration for performance profiling during model training and inference. |
| [`QuantizationConfig`](/docs/reference/wiki/deployment/quantizationconfig/) | Configuration for model quantization - compressing models by using lower precision numbers. |
| [`QuantizationConfiguration`](/docs/reference/wiki/deployment/quantizationconfiguration/) | Configuration for model quantization - comprehensive settings for PTQ and QAT. |
| [`RuntimeConfiguration`](/docs/reference/wiki/deployment/runtimeconfiguration/) | Configuration for the deployment runtime environment. |
| [`TFLiteConfiguration`](/docs/reference/wiki/deployment/tfliteconfiguration/) | Configuration for TensorFlow Lite model export. |
| [`TelemetryConfig`](/docs/reference/wiki/deployment/telemetryconfig/) | Configuration for telemetry - tracking and monitoring model inference metrics. |
| [`TensorRTConfiguration`](/docs/reference/wiki/deployment/tensorrtconfiguration/) | Configuration for TensorRT model conversion and execution. |
| [`VersioningConfig`](/docs/reference/wiki/deployment/versioningconfig/) | Configuration for model versioning - managing multiple versions of the same model. |
| [`WeightStreamingConfig`](/docs/reference/wiki/deployment/weightstreamingconfig/) | Configuration for weight streaming (paging large model weights to disk when they don't fit in RAM). |

## Helpers & Utilities (2)

| Type | Summary |
|:-----|:--------|
| [`CalibrationHelper<T, TInput, TOutput>`](/docs/reference/wiki/deployment/calibrationhelper/) | Helper class for calibrating quantizers using real forward passes. |
| [`OnnxNode`](/docs/reference/wiki/deployment/onnxnode/) | Represents an ONNX node (input or output). |


# YAML Configuration Reference

AiDotNet supports full YAML-based configuration for `AiModelBuilder`. Define your entire model training pipeline in a single YAML file -- no code changes needed.

## Quick Start

```yaml
# yaml-language-server: $schema=../../schemas/aidotnet-config.schema.json

optimizer:
  type: "Adam"

regularization:
  type: "NoRegularization"

preprocessing:
  steps:
    - type: "StandardScaler"
    - type: "SimpleImputer"
      params:
        strategy: "Mean"
```

```csharp
var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>("config.yaml");
var result = await builder.BuildAsync();
```

## Schema Autocomplete

For IntelliSense and validation in VS Code, add the schema directive at the top of your YAML file:

```yaml
# yaml-language-server: $schema=./schemas/aidotnet-config.schema.json
```

This requires the [YAML Language Server](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml) extension.

## Section Types

AiDotNet YAML configuration supports four section patterns:

### 1. Enum-Based Selection

Select a predefined algorithm by name. Maps to an enum value.

```yaml
optimizer:
  type: "Adam"

timeSeriesModel:
  type: "ARIMA"
```

### 2. Interface-Based (type + params)

Select a concrete implementation by class name and optionally set properties.

```yaml
regularization:
  type: "NoRegularization"

tokenizer:
  type: "BPETokenizer"
  params:
    vocabSize: 32000
```

### 3. POCO Configuration

Set properties directly on a configuration object. No `type` key needed.

```yaml
quantization:
  mode: "Int8"
  strategy: "Dynamic"
  granularity: "PerChannel"
  useSymmetricQuantization: true
```

### 4. Pipeline (steps)

Define ordered processing steps, each with a type and optional params.

```yaml
preprocessing:
  steps:
    - type: "StandardScaler"
    - type: "SimpleImputer"
      params:
        strategy: "Mean"
```

---

## Enum-Based Sections

### optimizer

Select the optimizer algorithm for model training.

```yaml
optimizer:
  type: "Adam"
```

**Available types:** Adam, SGD, RMSProp, AdaGrad, AdaDelta, Nadam, AMSGrad, RAdam, AdaBelief, Lookahead, LAMB, LARS, SWA, Ranger, and more.

See [Optimizers Reference](./optimizers.md) for full details.

### timeSeriesModel

Select the time series model type.

```yaml
timeSeriesModel:
  type: "ARIMA"
```

**Available types:** ARIMA, ExponentialSmoothing, HoltWinters, SARIMA, Prophet, VAR, GARCH, TBATS, and more.

---

## Deployment and Infrastructure

### quantization

Model quantization configuration for lower precision inference.

```yaml
quantization:
  mode: "Int8"
  strategy: "Dynamic"
  granularity: "PerChannel"
  useSymmetricQuantization: true
```

| Property | Type | Description |
|----------|------|-------------|
| `mode` | enum (QuantizationMode) | Quantization bit-width: Int8, Int4, Float16, etc. |
| `strategy` | enum (QuantizationStrategy) | Dynamic or Static quantization |
| `granularity` | enum (QuantizationGranularity) | PerTensor or PerChannel |
| `useSymmetricQuantization` | boolean | Use symmetric vs asymmetric ranges |

### compression

Model compression configuration for reducing model size.

```yaml
compression:
  enabled: true
  compressionLevel: 5
```

### caching

Model caching configuration for storing loaded models.

```yaml
caching:
  enabled: true
  maxCacheSize: 100
```

### abTesting

A/B testing configuration for comparing model versions.

```yaml
abTesting:
  enabled: true
```

### telemetry

Telemetry configuration for tracking inference metrics.

```yaml
telemetry:
  enabled: true
  trackLatency: true
  trackErrors: true
  samplingRate: 1.0
```

### mixedPrecision

Mixed precision training configuration.

```yaml
mixedPrecision:
  enabled: true
```

### inferenceOptimizations

Inference optimization configuration (KV caching, batching, speculative decoding).

```yaml
inferenceOptimizations:
  enableKVCaching: true
  enableBatching: true
  maxBatchSize: 32
```

### memoryManagement

Training memory management configuration.

```yaml
memoryManagement:
  enableGradientCheckpointing: true
```

---

## Interface-Based Sections

These sections use `type` to select a concrete implementation and `params` to configure it. Property names in `params` are case-insensitive.

### regularization

```yaml
regularization:
  type: "NoRegularization"
```

**Available types:** NoRegularization

### fitDetector

```yaml
fitDetector:
  type: "DefaultFitDetector"
```

**Available types:** DefaultFitDetector

### fairnessEvaluator

```yaml
fairnessEvaluator:
  type: "BasicFairnessEvaluator"
```

**Available types:** BasicFairnessEvaluator, ComprehensiveFairnessEvaluator, GroupFairnessEvaluator

### promptTemplate

```yaml
promptTemplate:
  type: "InstructionFollowingTemplate"
```

**Available types:** InstructionFollowingTemplate

### promptOptimizer

```yaml
promptOptimizer:
  type: "DiscreteSearchOptimizer"
```

**Available types:** DiscreteSearchOptimizer

### fewShotExampleSelector

```yaml
fewShotExampleSelector:
  type: "FixedExampleSelector"
```

**Available types:** FixedExampleSelector

### promptAnalyzer

```yaml
promptAnalyzer:
  type: "PatternDetectionAnalyzer"
```

**Available types:** PatternDetectionAnalyzer

### tokenizer

```yaml
tokenizer:
  type: "BPETokenizer"
  params:
    vocabSize: 32000
```

### trainingMonitor

```yaml
trainingMonitor:
  type: "TrainingMonitor"
```

---

## Pipeline Sections

### preprocessing

Define ordered preprocessing steps applied to your data before training.

```yaml
preprocessing:
  steps:
    - type: "StandardScaler"
    - type: "SimpleImputer"
      params:
        strategy: "Mean"
```

---

## Complete Example

See [examples/configs/ai-model-builder.yaml](https://github.com/ooples/AiDotNet/blob/master/examples/configs/ai-model-builder.yaml) for a comprehensive example covering all section types.

## Generating Documentation Programmatically

You can generate a complete reference document and JSON Schema at runtime:

```csharp
using AiDotNet.Configuration;

// Generate markdown documentation
var markdown = YamlDocsGenerator.Generate();
File.WriteAllText("yaml-config-reference.md", markdown);

// Generate JSON Schema for editor autocomplete
var jsonSchema = YamlJsonSchema.Generate();
File.WriteAllText("aidotnet-config.schema.json", jsonSchema);
```

## See Also

- [API Reference](../api/index.md) - Full API documentation
- [Getting Started](../getting-started/index.md) - Installation and first steps
- [Tutorials](../tutorials/index.md) - Step-by-step guides

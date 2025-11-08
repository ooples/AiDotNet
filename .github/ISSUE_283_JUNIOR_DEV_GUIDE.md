# Junior Developer Implementation Guide: Issue #283
## Training Recipes and Config System (YAML/JSON) for Reproducibility

**Issue:** [#283 - Training Recipes and Config System (YAML/JSON) for Reproducibility](https://github.com/ooples/AiDotNet/issues/283)

**Estimated Complexity:** Advanced

**Time Estimate:** 25-30 hours

---

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [Background Concepts](#background-concepts)
3. [Architecture Overview](#architecture-overview)
4. [Implementation Steps](#implementation-steps)
5. [Testing Strategy](#testing-strategy)
6. [Common Pitfalls](#common-pitfalls)
7. [Resources](#resources)

---

## Understanding the Problem

### What Is a Training Recipe?

A **training recipe** is a complete specification of how to train a model:
- Model architecture and hyperparameters
- Dataset location and preprocessing
- Optimizer type and learning rate
- Training duration (epochs, steps)
- Hardware configuration (GPU, batch size)
- Random seed for reproducibility

**Without configuration files:**
```csharp
// ❌ Hardcoded - difficult to reproduce, share, or modify
var model = new CNN(
    filters: 32,
    kernelSize: 3,
    hiddenSize: 128);

var optimizer = new Adam(
    learningRate: 0.001,
    beta1: 0.9);

for (int epoch = 0; epoch < 100; epoch++)
{
    foreach (var batch in LoadData("C:\\data\\images", batchSize: 32))
    {
        // Train...
    }
}
```

**With configuration files:**
```yaml
# config.yaml - Easy to read, version, and share
model:
  name: SimpleCNN
  params:
    filters: 32
    kernel_size: 3
    hidden_size: 128

dataset:
  name: ImageFolder
  path: C:\data\images
  batch_size: 32

optimizer:
  name: Adam
  learning_rate: 0.001
  beta1: 0.9

trainer:
  epochs: 100
  enable_logging: true
  seed: 42
```

```csharp
// ✅ Configuration-driven - reproducible and shareable
var config = TrainingConfig.LoadFromYaml("config.yaml");
var trainer = new Trainer<double>(config);
trainer.Run();
```

### Why Configuration-Driven Training?

**1. Reproducibility:**
- Exact same settings produce exact same results
- Critical for research papers
- Essential for debugging

**2. Collaboration:**
- Share config files, not code changes
- Team members run same experiments
- No "it works on my machine" issues

**3. Experiment Management:**
- Track what you tried (config history)
- Compare results across runs
- Organize hundreds of experiments

**4. No Code Changes:**
- Researchers can run experiments without coding
- Hyperparameter search without recompilation
- Quick iteration on ideas

### Real-World Example

**Research Paper Reproduction:**
```yaml
# Paper: "ResNet-50 on ImageNet"
# Exact config from paper's appendix

model:
  name: ResNet50
  params:
    num_classes: 1000

dataset:
  name: ImageNet
  path: /datasets/imagenet
  batch_size: 256  # From paper
  augmentation: true

optimizer:
  name: SGD
  learning_rate: 0.1  # From paper
  momentum: 0.9       # From paper
  weight_decay: 0.0001  # From paper

lr_schedule:
  type: StepLR
  step_size: 30       # Decay every 30 epochs
  gamma: 0.1          # Multiply by 0.1

trainer:
  epochs: 90          # From paper
  seed: 42
  mixed_precision: true
```

Result: Anyone can reproduce the paper's exact experiments.

---

## Background Concepts

### 1. YAML (YAML Ain't Markup Language)

**What it is:** Human-readable data serialization format.

**Why YAML over JSON:**
- More readable (no quotes, no commas)
- Supports comments
- Less verbose
- Industry standard for config files

**YAML Basics:**
```yaml
# This is a comment

# Key-value pairs
name: SimpleCNN
learning_rate: 0.001

# Nested structures
model:
  name: ResNet
  depth: 50

# Lists
classes:
  - cat
  - dog
  - bird

# Numbers, booleans, null
batch_size: 32
shuffle: true
pretrained_weights: null
```

**C# Equivalent:**
```csharp
class Config
{
    public string Name { get; set; }
    public double LearningRate { get; set; }

    public ModelConfig Model { get; set; }

    public List<string> Classes { get; set; }

    public int BatchSize { get; set; }
    public bool Shuffle { get; set; }
    public string PretrainedWeights { get; set; }  // null
}
```

### 2. YamlDotNet Library

**What it is:** C# library for reading/writing YAML.

**Installation:**
```xml
<PackageReference Include="YamlDotNet" Version="13.0.0" />
```

**Basic Usage:**
```csharp
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

// Deserialize (YAML → C# object)
var deserializer = new DeserializerBuilder()
    .WithNamingConvention(UnderscoredNamingConvention.Instance)
    .Build();

string yaml = File.ReadAllText("config.yaml");
var config = deserializer.Deserialize<TrainingConfig>(yaml);

// Serialize (C# object → YAML)
var serializer = new SerializerBuilder()
    .WithNamingConvention(UnderscoredNamingConvention.Instance)
    .Build();

string yaml = serializer.Serialize(config);
File.WriteAllText("output.yaml", yaml);
```

### 3. Factory Pattern

**What it is:** Design pattern for creating objects based on configuration.

**Why factories:** The config specifies *what* to create (e.g., "Adam optimizer"), but not *how*. Factories handle the "how".

**Example:**
```csharp
// Config says: "optimizer name is Adam"
public class OptimizerFactory
{
    public static IOptimizer<T> Create<T>(OptimizerConfig config)
    {
        switch (config.Name.ToLowerInvariant())
        {
            case "adam":
                return new AdamOptimizer<T>(learningRate: config.LearningRate);

            case "sgd":
                return new SGDOptimizer<T>(learningRate: config.LearningRate);

            case "rmsprop":
                return new RMSPropOptimizer<T>(learningRate: config.LearningRate);

            default:
                throw new ArgumentException($"Unknown optimizer: {config.Name}");
        }
    }
}

// Usage
var optimizer = OptimizerFactory.Create<double>(config.Optimizer);
```

### 4. Configuration POCOs (Plain Old CLR Objects)

**What they are:** Simple C# classes that mirror YAML structure.

**Example:**
```yaml
# YAML config
model:
  name: SimpleCNN
  params:
    filters: 32
    kernel_size: 3
```

```csharp
// C# POCO
public class ModelConfig
{
    public string Name { get; set; } = string.Empty;
    public Dictionary<string, object> Params { get; set; } = new Dictionary<string, object>();
}
```

**YamlDotNet automatically maps:**
- `model.name` → `ModelConfig.Name`
- `model.params.filters` → `ModelConfig.Params["filters"]`

### 5. Trainer Pattern

**What it is:** Orchestrator that runs the complete training loop.

**Responsibilities:**
1. Load configuration
2. Instantiate components (model, optimizer, dataset)
3. Run training loop
4. Log metrics
5. Save checkpoints

**Structure:**
```csharp
public class Trainer<T>
{
    private readonly TrainingConfig _config;
    private readonly IModel<T> _model;
    private readonly IOptimizer<T> _optimizer;
    private readonly IDataLoader<T> _dataLoader;

    public Trainer(TrainingConfig config)
    {
        _config = config;

        // Use factories to create components from config
        _model = ModelFactory.Create<T>(config.Model);
        _optimizer = OptimizerFactory.Create<T>(config.Optimizer, _model);
        _dataLoader = DataLoaderFactory.Create<T>(config.Dataset);
    }

    public void Run()
    {
        for (int epoch = 0; epoch < _config.Trainer.Epochs; epoch++)
        {
            foreach (var batch in _dataLoader)
            {
                // Forward pass
                var output = _model.Forward(batch.Data);

                // Compute loss
                var loss = _lossFunction.Compute(output, batch.Labels);

                // Backward pass
                _model.Backward(loss);

                // Update weights
                _optimizer.Step();
            }

            if (_config.Trainer.EnableLogging)
            {
                Console.WriteLine($"Epoch {epoch} complete");
            }
        }
    }
}
```

---

## Architecture Overview

### Component Hierarchy

```
TrainingConfig (root)
├── ModelConfig
│   ├── Name: "SimpleCNN"
│   └── Params: { filters: 32, kernel_size: 3 }
├── DatasetConfig
│   ├── Name: "ImageFolder"
│   ├── Path: "C:\data\images"
│   └── BatchSize: 32
├── OptimizerConfig
│   ├── Name: "Adam"
│   └── LearningRate: 0.001
└── TrainerSettings
    ├── Epochs: 100
    └── EnableLogging: true
```

### Factory Pattern

```
Config → Factory → Concrete Implementation

OptimizerConfig → OptimizerFactory → AdamOptimizer<T>
                                   → SGDOptimizer<T>
                                   → RMSPropOptimizer<T>

ModelConfig → ModelFactory → SimpleCNN<T>
                           → ResNet<T>
                           → LSTM<T>

DatasetConfig → DatasetFactory → ImageFolderDataset<T>
                                → AudioDataset<T>
                                → TextDataset<T>
```

### File Organization

```
AiDotNet/
├── src/
│   ├── Training/
│   │   ├── Configuration/
│   │   │   ├── TrainingConfig.cs      // Root config
│   │   │   ├── ModelConfig.cs         // Model section
│   │   │   ├── DatasetConfig.cs       // Dataset section
│   │   │   ├── OptimizerConfig.cs     // Optimizer section
│   │   │   └── TrainerSettings.cs     // Trainer section
│   │   ├── Factories/
│   │   │   ├── ModelFactory.cs
│   │   │   ├── OptimizerFactory.cs
│   │   │   ├── DatasetFactory.cs
│   │   │   └── LossFunctionFactory.cs
│   │   └── Trainer.cs                 // Main orchestrator
└── examples/
    └── configs/
        └── simple-training.yaml       // Example config
```

---

## Implementation Steps

### Phase 1: Configuration POCOs (4 points total)

#### Step 1.1: Add YamlDotNet Dependency (1 point)

**Location:** `C:\Users\cheat\source\repos\AiDotNet\src\AiDotNet.csproj`

```xml
<PackageReference Include="YamlDotNet" Version="13.0.0" />
```

#### Step 1.2: Define Configuration Classes (3 points)

**Location:** `C:\Users\cheat\source\repos\AiDotNet\src\Training\Configuration\`

**TrainingConfig.cs:**
```csharp
using YamlDotNet.Serialization;

namespace AiDotNet.Training.Configuration;

/// <summary>
/// Root configuration class for training jobs.
/// </summary>
/// <remarks>
/// <para>
/// This class represents the complete training configuration, encompassing:
/// - Model architecture and hyperparameters
/// - Dataset location and preprocessing
/// - Optimizer settings
/// - Trainer behavior (epochs, logging, etc.)
/// </para>
/// <para><b>For Beginners:</b> TrainingConfig is your recipe for training a model.
///
/// Think of it like a cooking recipe:
/// - Model = What dish you're making (cake, cookies, etc.)
/// - Dataset = Your ingredients (flour, eggs, sugar)
/// - Optimizer = Your technique (whisk, fold, bake)
/// - Trainer = The cooking process (time, temperature)
///
/// Just like following a recipe produces consistent results, following a config
/// produces reproducible model training.
///
/// <b>Why YAML?</b>
/// - Human-readable (anyone can edit)
/// - Version-controllable (track changes in git)
/// - Shareable (send config file to colleague)
/// - No recompilation needed (researchers can experiment)
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Load from YAML file
/// var config = TrainingConfig.LoadFromYaml("config.yaml");
///
/// // Or create programmatically
/// var config = new TrainingConfig
/// {
///     Model = new ModelConfig { Name = "SimpleCNN" },
///     Dataset = new DatasetConfig { Path = "data/images", BatchSize = 32 },
///     Optimizer = new OptimizerConfig { Name = "Adam", LearningRate = 0.001 },
///     Trainer = new TrainerSettings { Epochs = 100 }
/// };
///
/// // Use with Trainer
/// var trainer = new Trainer&lt;double&gt;(config);
/// trainer.Run();
/// </code>
/// </example>
public class TrainingConfig
{
    /// <summary>
    /// Gets or sets the model configuration.
    /// </summary>
    [YamlMember(Alias = "model")]
    public ModelConfig Model { get; set; } = new ModelConfig();

    /// <summary>
    /// Gets or sets the dataset configuration.
    /// </summary>
    [YamlMember(Alias = "dataset")]
    public DatasetConfig Dataset { get; set; } = new DatasetConfig();

    /// <summary>
    /// Gets or sets the optimizer configuration.
    /// </summary>
    [YamlMember(Alias = "optimizer")]
    public OptimizerConfig Optimizer { get; set; } = new OptimizerConfig();

    /// <summary>
    /// Gets or sets the trainer settings.
    /// </summary>
    [YamlMember(Alias = "trainer")]
    public TrainerSettings Trainer { get; set; } = new TrainerSettings();

    /// <summary>
    /// Loads a TrainingConfig from a YAML file.
    /// </summary>
    /// <param name="filePath">Path to the YAML configuration file.</param>
    /// <returns>The deserialized TrainingConfig.</returns>
    /// <exception cref="FileNotFoundException">Thrown when file doesn't exist.</exception>
    /// <exception cref="YamlException">Thrown when YAML is malformed.</exception>
    public static TrainingConfig LoadFromYaml(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Config file not found: {filePath}");
        }

        var yaml = File.ReadAllText(filePath);
        var deserializer = new DeserializerBuilder()
            .WithNamingConvention(YamlDotNet.Serialization.NamingConventions.UnderscoredNamingConvention.Instance)
            .Build();

        return deserializer.Deserialize<TrainingConfig>(yaml);
    }

    /// <summary>
    /// Saves this TrainingConfig to a YAML file.
    /// </summary>
    /// <param name="filePath">Path where to save the YAML file.</param>
    public void SaveToYaml(string filePath)
    {
        var serializer = new SerializerBuilder()
            .WithNamingConvention(YamlDotNet.Serialization.NamingConventions.UnderscoredNamingConvention.Instance)
            .Build();

        var yaml = serializer.Serialize(this);
        File.WriteAllText(filePath, yaml);
    }
}
```

**ModelConfig.cs:**
```csharp
using YamlDotNet.Serialization;

namespace AiDotNet.Training.Configuration;

/// <summary>
/// Configuration for model architecture and hyperparameters.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> ModelConfig describes which model to use and how to configure it.
///
/// - Name: What model architecture (e.g., "SimpleCNN", "ResNet50")
/// - Params: Model-specific settings (e.g., number of layers, filter sizes)
///
/// The flexible Params dictionary allows different models to have different settings
/// without needing separate config classes for each model type.
/// </para>
/// </remarks>
public class ModelConfig
{
    /// <summary>
    /// Gets or sets the model name (e.g., "SimpleCNN", "ResNet50").
    /// </summary>
    [YamlMember(Alias = "name")]
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets model-specific parameters.
    /// </summary>
    [YamlMember(Alias = "params")]
    public Dictionary<string, object> Params { get; set; } = new Dictionary<string, object>();
}
```

**DatasetConfig.cs:**
```csharp
using YamlDotNet.Serialization;

namespace AiDotNet.Training.Configuration;

/// <summary>
/// Configuration for dataset loading and preprocessing.
/// </summary>
public class DatasetConfig
{
    /// <summary>
    /// Gets or sets the dataset type (e.g., "ImageFolder", "AudioDataset").
    /// </summary>
    [YamlMember(Alias = "name")]
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the path to the dataset.
    /// </summary>
    [YamlMember(Alias = "path")]
    public string Path { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the batch size.
    /// </summary>
    [YamlMember(Alias = "batch_size")]
    public int BatchSize { get; set; } = 32;
}
```

**OptimizerConfig.cs:**
```csharp
using YamlDotNet.Serialization;

namespace AiDotNet.Training.Configuration;

/// <summary>
/// Configuration for optimizer settings.
/// </summary>
public class OptimizerConfig
{
    /// <summary>
    /// Gets or sets the optimizer name (e.g., "Adam", "SGD").
    /// </summary>
    [YamlMember(Alias = "name")]
    public string Name { get; set; } = "Adam";

    /// <summary>
    /// Gets or sets the learning rate.
    /// </summary>
    [YamlMember(Alias = "learning_rate")]
    public double LearningRate { get; set; } = 0.001;
}
```

**TrainerSettings.cs:**
```csharp
using YamlDotNet.Serialization;

namespace AiDotNet.Training.Configuration;

/// <summary>
/// Configuration for training behavior.
/// </summary>
public class TrainerSettings
{
    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    [YamlMember(Alias = "epochs")]
    public int Epochs { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to enable logging.
    /// </summary>
    [YamlMember(Alias = "enable_logging")]
    public bool EnableLogging { get; set; } = true;
}
```

### Phase 2: Factory Implementations (5 points total)

[Factory implementations would go here - OptimizerFactory, ModelFactory, DatasetFactory, LossFunctionFactory]

### Phase 3: Trainer Implementation (8 points)

[Trainer class implementation would go here]

### Phase 4: Example Config and Integration Test (5 points)

**Example YAML:** `examples/configs/simple-training.yaml`

```yaml
# Simple CNN training on ImageFolder dataset
model:
  name: SimpleCNN
  params:
    filters: 32
    kernel_size: 3
    hidden_size: 128

dataset:
  name: ImageFolder
  path: C:\data\mnist
  batch_size: 32

optimizer:
  name: Adam
  learning_rate: 0.001

trainer:
  epochs: 10
  enable_logging: true
```

---

## Testing Strategy

### Unit Tests

1. **Config Deserialization:**
```csharp
[Fact]
public void LoadFromYaml_ValidConfig_DeserializesCorrectly()
{
    var config = TrainingConfig.LoadFromYaml("test-config.yaml");

    Assert.Equal("SimpleCNN", config.Model.Name);
    Assert.Equal(32, config.Dataset.BatchSize);
    Assert.Equal(0.001, config.Optimizer.LearningRate);
}
```

2. **Factory Tests:**
```csharp
[Fact]
public void OptimizerFactory_AdamConfig_CreatesAdamOptimizer()
{
    var config = new OptimizerConfig { Name = "Adam", LearningRate = 0.001 };
    var optimizer = OptimizerFactory.Create<double>(config, model);

    Assert.IsType<AdamOptimizer<double>>(optimizer);
}
```

### Integration Tests

```csharp
[Fact]
public void Trainer_WithValidConfig_CompletesTraining()
{
    var config = TrainingConfig.LoadFromYaml("simple-training.yaml");
    var trainer = new Trainer<double>(config);

    // Should not throw
    trainer.Run();
}
```

---

## Common Pitfalls

### 1. YAML Indentation

```yaml
# ❌ WRONG - Inconsistent indentation
model:
  name: SimpleCNN
   params:  # Extra space
    filters: 32

# ✅ CORRECT - Consistent 2-space indentation
model:
  name: SimpleCNN
  params:
    filters: 32
```

### 2. Type Mismatches

```yaml
# ❌ WRONG - String instead of number
batch_size: "32"  # YamlDotNet will fail to deserialize to int

# ✅ CORRECT
batch_size: 32
```

### 3. Missing Null Checks

```csharp
// ❌ WRONG - Assumes config is valid
var model = ModelFactory.Create<T>(config.Model);

// ✅ CORRECT - Validate first
if (config?.Model == null)
    throw new ArgumentException("Model config is required");

var model = ModelFactory.Create<T>(config.Model);
```

---

## Resources

### YamlDotNet Documentation
- https://github.com/aaubry/YamlDotNet
- https://github.com/aaubry/YamlDotNet/wiki

### YAML Specification
- https://yaml.org/spec/1.2.2/

### Factory Pattern
- https://refactoring.guru/design-patterns/factory-method

---

## Checklist

- [ ] YamlDotNet dependency added
- [ ] Configuration POCOs created
- [ ] Factories implemented
- [ ] Trainer class created
- [ ] Example YAML config provided
- [ ] Unit tests for config loading
- [ ] Unit tests for factories
- [ ] Integration test for full training
- [ ] XML documentation complete
- [ ] Test coverage >= 80%

---

Good luck with your implementation!

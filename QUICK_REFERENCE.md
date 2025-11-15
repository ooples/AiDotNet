# AiDotNet Quick Reference Guide

## Project Statistics
- **Total C# Files in src/:** 1,158
- **Directories in src/:** 43
- **Neural Network Types:** 44 (LSTM, GAN, CNN, Transformer, etc.)
- **Optimizer Types:** 40+
- **Configuration Option Classes:** 170+
- **Interface Definitions:** 75+
- **Enum Types:** 60+

## Directory Organization Map

```
src/
├── [Core ML] NeuralNetworks/, Optimizers/, LossFunctions/, ActivationFunctions/
├── [Training] MetaLearning/, AutoML/, Evaluation/, FitDetectors/
├── [Models] Models/ (Metadata, Options, Results, Inputs)
├── [Data] Data/, DataProcessor/, Normalizers/, FeatureSelectors/
├── [Infrastructure] Serialization/, Caching/, Interfaces/, Enums/, Exceptions/
├── [Advanced] LoRA/, TransferLearning/, RetrievalAugmentedGeneration/
├── [Utilities] LinearAlgebra/, Statistics/, Helpers/, Extensions/
└── [Specialized] TimeSeries/, Regression/, Interpretation/, Images/
```

## Key File Locations

| Purpose | Location | Key Classes |
|---------|----------|-------------|
| Model Training | `/Optimizers/` | `OptimizerBase<T, TInput, TOutput>`, 40+ optimizer classes |
| Meta-Learning | `/MetaLearning/` | `ReptileTrainer`, `ReptileTrainerConfig` |
| AutoML | `/AutoML/` | `AutoMLModelBase`, `SuperNet`, `TrialResult` |
| Model Definition | `/Models/` | `ModelMetadata<T>`, `NeuralNetworkModel<T>` |
| Evaluation | `/Evaluation/` | `DefaultModelEvaluator<T, TInput, TOutput>` |
| Serialization | `/Serialization/` | JSON converters for Tensors, Matrices, Vectors |
| Interfaces | `/Interfaces/` | 75 core interfaces (must-read for architecture) |
| Configuration | `/Models/Options/` | 170+ option classes for all algorithms |
| Results | `/Models/Results/` | OptimizationResult, MetaTrainingResult, etc. |

## Naming Conventions at a Glance

```
Optimizers:        {Algorithm}Optimizer (AdamOptimizer)
Options/Config:    {Algorithm}Options (AdamOptimizerOptions)
Neural Networks:   {Type}NeuralNetwork (LSTMNeuralNetwork)
Trainers:          {Type}Trainer + {Type}TrainerConfig
Result Classes:    {Operation}Result (OptimizationResult)
Data Loaders:      {Strategy}EpisodicDataLoader
Exceptions:        {Type}Exception (ModelTrainingException)
Interfaces:        I{Name} (IOptimizer, IModel)
```

## Architecture Patterns

**Generic Type Parameters Everywhere:**
- `T` = numeric type (float, double)
- `TInput` = input data type
- `TOutput` = output/target data type
- Example: `OptimizerBase<T, TInput, TOutput>`

**Three-Tier Structure:**
1. Interfaces (`/Interfaces/`) → Define contracts
2. Base Classes (`*Base.cs`) → Provide common implementation
3. Implementations (`specific algorithms`)

**Builder Pattern:**
- `PredictionModelBuilder<T, TInput, TOutput>` - Main builder
- Fluent API for configuration

## Current Capabilities

### Supported
- 40+ optimization algorithms
- 44 neural network architectures
- Meta-learning (Reptile)
- AutoML with trial-based search
- Model evaluation and fit detection
- Data normalization and feature selection
- Transfer learning and LoRA
- RAG (Retrieval Augmented Generation)
- Cross-validation strategies

### NOT Currently Supported
- Experiment tracking/MLOps
- Model checkpointing
- Training monitoring/logging
- Model registry
- Advanced data versioning
- Distributed training
- Training callbacks/hooks

## Dependencies
- **Newtonsoft.Json** - Serialization
- **Azure.Search.Documents** - Document storage
- **Elasticsearch** - Document search
- **Pinecone.Client** - Vector DB
- **StackExchange.Redis** - Caching
- **Npgsql.EntityFrameworkCore.PostgreSQL** - Database (.NET 8 only)

## Code Style Highlights

1. **Extensive XML Documentation** - Every public class/method documented
2. **Beginner-Friendly Comments** - "For Beginners" sections explain concepts
3. **Interface-First Design** - Contracts before implementations
4. **Validation Methods** - `IsValid()` on configuration classes
5. **Configuration Options** - Every algorithm has an Options class
6. **Global Usings** - Uses file-scoped namespaces
7. **Nullable Enabled** - Strict null checking enabled

## Testing Structure
- `/tests/` - Unit tests organized by feature area
- `/AiDotNetBenchmarkTests/` - Performance benchmarks
- `/testconsole/` - Example applications

## Recommended Reading Order (for new contributors)

1. **Start:** `/Interfaces/IModel.cs` - Core concept
2. **Next:** `/Interfaces/IOptimizer.cs` - Training concept
3. **Then:** `/Optimizers/OptimizerBase.cs` - Base implementation
4. **Study:** `/MetaLearning/Config/ReptileTrainerConfig.cs` - Config pattern
5. **Review:** `/Models/ModelMetadata.cs` - Metadata pattern
6. **Analyze:** `/AutoML/AutoMLModelBase.cs` - Complex integration

## Integration Points for New Components

1. **Hook into OptimizerBase:**
   - Override iteration methods
   - Use `IterationHistoryList` for tracking
   - Leverage `ModelCache` for caching

2. **Use ModelMetadata:**
   - Extend properties dictionary
   - Store custom information

3. **Integrate with PredictionModelBuilder:**
   - Add Configure* methods
   - Support builder chaining

4. **Follow Options Pattern:**
   - Create {Feature}Options in Models/Options/
   - Implement IsValid() for validation

5. **Use Enums:**
   - Add new enums to Enums/ for type safety
   - Follow naming: {FeatureName}{Enum}

## File Size Reference
- Largest: `NeuralNetworkBase.cs` (79 KB)
- Very Large: `DifferentiableNeuralComputer.cs` (95 KB)
- Typical: 2-30 KB per file
- Options classes: 5-50 KB

## Build Configuration
- **Target Frameworks:** net8.0, net462 (dual targeting)
- **Language Version:** latest
- **Nullable:** enabled
- **Warnings as Errors:** true (both Debug and Release)
- **NuGet Package:** Published with auto-build enabled

---

**For the ML Training Infrastructure Implementation:**
See `/CODEBASE_ANALYSIS.md` for detailed recommendations on:
- Where to place 6 new components
- How to structure directories
- Integration points
- Implementation order

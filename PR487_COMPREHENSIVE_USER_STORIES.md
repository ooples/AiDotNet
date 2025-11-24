# PR#487 JIT Compilation - Comprehensive User Stories (CORRECTED)

**Based on**: PR487_CORRECTED_COMPREHENSIVE_GAP_ANALYSIS.md

---

## Table of Contents
1. [Critical Architectural Fixes](#critical-architectural-fixes)
2. [Base Class Infrastructure](#base-class-infrastructure)
3. [TensorOperations IEngine Overhaul](#tensoroperations-iengine-overhaul)
4. [Layer Implementations (65 Remaining)](#layer-implementations-65-remaining)
5. [Model Implementations (115 Models)](#model-implementations-115-models)
6. [Integration & Testing](#integration-testing)
7. [Estimation Summary](#estimation-summary)

---

## Critical Architectural Fixes

**THESE MUST BE DONE FIRST IN ORDER**

### US-ARCH-1: Make LayerBase JIT Methods Abstract
**As a** layer developer
**I need** LayerBase to enforce JIT implementation with abstract methods
**So that** all 75 layers are forced to implement ExportComputationGraph and SupportsJitCompilation

**Current State** (src/NeuralNetworks/Layers/LayerBase.cs:702, 729):
```csharp
public virtual ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
{
    throw new NotImplementedException("Override ExportComputationGraph()...");
}
public virtual bool SupportsJitCompilation => false;
```

**Problem**: VIRTUAL allows layers to skip implementation

**Solution**:
```csharp
public abstract ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes);
public abstract bool SupportsJitCompilation { get; }
```

**Acceptance Criteria**:
- LayerBase.cs lines 702 and 729 changed from `virtual` to `abstract`
- Remove method body from ExportComputationGraph
- Remove default return from SupportsJitCompilation
- All 75 layers must now implement both members (build will fail until done)
- Update LayerBase documentation

**Estimated Effort**: 1-2 hours
**Priority**: CRITICAL - BLOCKS ALL OTHER WORK
**Files**: `src/NeuralNetworks/Layers/LayerBase.cs`

**Impact**: After this change, all 75 layers will have compile errors until they implement both methods

---

### US-ARCH-2: Remove Open/Closed Violations from NeuralNetworkBase
**As a** maintainer
**I need** to remove 900+ lines of layer-specific code from NeuralNetworkBase
**So that** the base class follows SOLID principles and new layers don't require base class modifications

**Current State** (src/NeuralNetworks/NeuralNetworkBase.cs:2405-3513):
- `ConvertLayerToGraph()` - 82 lines with giant switch statement
- 40+ `Convert*Layer()` private methods - 900+ lines total
- Every new layer requires modifying NeuralNetworkBase

**Problem**: VIOLATES OPEN/CLOSED PRINCIPLE
- ConvertDenseLayer (45 lines)
- ConvertFullyConnectedLayer (48 lines)
- ConvertFeedForwardLayer (34 lines)
- ConvertActivationLayer (20 lines)
- ConvertBatchNormalizationLayer (69 lines)
- ConvertLayerNormalizationLayer (39 lines)
- ConvertResidualLayer (27 lines)
- ConvertConvolutionalLayer (23 lines)
- ConvertDeconvolutionalLayer (23 lines)
- ConvertLSTMLayer (38 lines)
- ConvertGRULayer (36 lines)
- ConvertAttentionLayer (32 lines)
- ConvertSelfAttentionLayer (32 lines)
- ConvertMultiHeadAttentionLayer (50 lines)
- ... 26 more Convert*Layer methods

**Solution**: DELETE ALL, delegate to layer.ExportComputationGraph()

**Correct Implementation**:
```csharp
public virtual ComputationNode<T> ExportComputationGraph()
{
    if (_layers.Count == 0)
        throw new InvalidOperationException("Cannot export computation graph: network has no layers");

    var inputNodes = new List<ComputationNode<T>>();
    var currentNode = _layers[0].ExportComputationGraph(inputNodes);

    for (int i = 1; i < _layers.Count; i++)
    {
        var layerInputs = new List<ComputationNode<T>> { currentNode };
        currentNode = _layers[i].ExportComputationGraph(layerInputs);
    }

    return currentNode;
}

public virtual bool SupportsJitCompilation
{
    get => _layers.Count > 0 && _layers.All(layer => layer.SupportsJitCompilation);
}
```

**Acceptance Criteria**:
- DELETE lines 2405-3513 from NeuralNetworkBase.cs (~1100 lines)
- Replace with simple loop (12 lines as shown above)
- All unit tests pass
- New layers can be added without touching NeuralNetworkBase
- Code review confirms SOLID principles restored

**Estimated Effort**: 4-8 hours
**Priority**: CRITICAL - MUST BE DONE BEFORE LAYER WORK
**Files**: `src/NeuralNetworks/NeuralNetworkBase.cs`

**Dependencies**: US-ARCH-1 must be complete first (layers need abstract methods)

---

### US-ARCH-3: Audit All Models for IFullModel Implementation
**As a** planning lead
**I need** complete audit of which models implement IFullModel
**So that** I can plan model JIT work accurately

**Current State**: IFullModel already has IJitCompilable<T> ✅ (src/Interfaces/IFullModel.cs:45)

**Problem**: Unknown which of 115 models actually implement IFullModel

**Solution**: Create comprehensive audit spreadsheet

**Acceptance Criteria**:
- Audit all 38 regression models
- Audit all 20 time series models
- Audit all 38 neural network models
- Audit all 10 ensemble models
- Audit all 9 distributed models
- Document format: `MODEL_IFULLMODEL_AUDIT.md` with columns:
  - Model Name
  - File Path
  - Implements IFullModel? (Yes/No/Unknown)
  - Has ExportComputationGraph? (Yes/No/Stub/N/A)
  - Priority for JIT (Critical/High/Medium/Low/Defer)
  - Notes

**Estimated Effort**: 3-5 hours
**Priority**: HIGH (needed for accurate planning)
**Deliverable**: `MODEL_IFULLMODEL_AUDIT.md`

---

## Base Class Infrastructure

### US-BASE-1: Implement NeuralNetworkBase JIT Support
**As a** neural network model implementer
**I need** NeuralNetworkBase to provide correct JIT implementation
**So that** all 38 neural network models get JIT support by inheritance

**Dependencies**:
- ✅ US-ARCH-1 complete (LayerBase abstract methods)
- ✅ US-ARCH-2 complete (Convert*Layer violations removed)

**Implementation** (already shown in US-ARCH-2):
- Simple delegation to layer.ExportComputationGraph()
- Check all layers support JIT

**Acceptance Criteria**:
- NeuralNetworkBase.ExportComputationGraph() chains all layers
- SupportsJitCompilation returns true only if ALL layers support it
- Works with any layer combination
- Unit tests with various architectures (CNN, RNN, Transformer)
- Integration tests with JitCompiler.Compile()

**Estimated Effort**: 2-4 hours (implementation is simple, testing takes time)
**Priority**: CRITICAL
**Files**: `src/NeuralNetworks/NeuralNetworkBase.cs`

**Key Insight**: Most neural network models will get JIT for FREE once this is done!

---

## TensorOperations IEngine Overhaul

### US-ENGINE-1: Audit All TensorOperations for IEngine Usage
**As a** performance engineer
**I need** complete audit of all 43+ TensorOperations methods
**So that** I know which operations need IEngine acceleration

**Current State**:
- Only 3/43+ operations use IEngine:
  - Add (line 132): `var engine = AiDotNetEngine.Current;`
  - MatrixMultiply (line 901): `var engine = AiDotNetEngine.Current;`
  - Transpose (line 968): `var engine = AiDotNetEngine.Current;`
- All others use direct Tensor methods or scalar loops

**Solution**: Create audit spreadsheet

**Acceptance Criteria**:
- List all 43+ TensorOperations methods
- Columns:
  - Operation Name
  - Line Number
  - Uses IEngine? (Yes/No)
  - Current Implementation (Direct/ScalarLoop/Mixed)
  - IEngine Method Needed
  - Priority (High/Medium/Low)
- Document: `TENSOROPERATIONS_IENGINE_AUDIT.md`

**Operations to Audit**:
1. Variable (line 72)
2. Constant (line 103)
3. Add (line 129) ✅ Uses IEngine
4. Subtract (line 206) ❌ Direct: `a.Value.ElementwiseSubtract(b.Value)`
5. ElementwiseMultiply (line 281) ❌ Direct: `a.Value.ElementwiseMultiply(b.Value)`
6. Divide (line 356) ❌ Direct: `a.Value.ElementwiseDivide(b.Value)`
7. Power (line 446) ❌ ScalarLoop: `a.Value.Transform((x, _) => numOps.Power(x, expValue))`
8. Exp (line 508) ❌ ScalarLoop: `a.Value.Transform((x, _) => numOps.Exp(x))`
9. Log (line 562) ❌ ScalarLoop
10. Sqrt (line 619) ❌ ScalarLoop
11. Tanh (line 679) ❌ ScalarLoop
12. Sigmoid (line 735) ❌ ScalarLoop
13. ReLU (line 794) ❌ ScalarLoop
14. Negate (line 849) ❌ ScalarLoop
15. MatrixMultiply (line 899) ✅ Uses IEngine
16. Transpose (line 966) ✅ Uses IEngine
17. Sum (line 1016) ❌ Complex nested loops
18. Mean (line 1142) ❌ Uses Sum
19. Reshape (line 1202) ❌ Direct
20. Softmax (line 1262) ❌ Complex: Exp + Sum + Divide
21. Concat (line 1379) ❌ Complex array copying
22. Pad (line 1567) ❌ Complex array copying
23. MaxPool2D (line 1675) ❌ Nested loops
24. AvgPool2D (line 1824) ❌ Nested loops
25. LayerNorm (line 1964) ❌ Complex: Mean + StdDev + normalize
26. BatchNorm (line 2191) ❌ Complex: Mean + Var + normalize
27. Conv2D (line 2474) ❌ Complex nested loops
28. ConvTranspose2D (line 2728) ❌ Complex nested loops
29. ReduceMax (line 2962) ❌ Nested loops
30. ReduceMean (line 3076) ❌ Nested loops
... (more operations)

**Estimated Effort**: 4-6 hours
**Priority**: HIGH
**Deliverable**: `TENSOROPERATIONS_IENGINE_AUDIT.md`

---

### US-ENGINE-2: Implement Missing IEngine Methods
**As a** library maintainer
**I need** IEngine to provide acceleration for all TensorOperations
**So that** CPU and GPU backends can optimize all operations

**Dependencies**: US-ENGINE-1 complete (audit tells us what's needed)

**Acceptance Criteria**:
- IEngine interface extended with methods for all operations
- CpuEngine implements all new methods
- GpuEngine implements all new methods (with GPU acceleration)
- Unit tests for each engine method

**New Methods Needed** (examples):
```csharp
public interface IEngine
{
    // Existing
    Tensor<T> TensorAdd<T>(Tensor<T> a, Tensor<T> b);
    Tensor<T> MatMul<T>(Tensor<T> a, Tensor<T> b);
    Tensor<T> Transpose<T>(Tensor<T> tensor, int[] axes);

    // NEW: Element-wise
    Tensor<T> TensorSubtract<T>(Tensor<T> a, Tensor<T> b);
    Tensor<T> TensorMultiply<T>(Tensor<T> a, Tensor<T> b);
    Tensor<T> TensorDivide<T>(Tensor<T> a, Tensor<T> b);
    Tensor<T> TensorPower<T>(Tensor<T> a, T exponent);
    Tensor<T> TensorExp<T>(Tensor<T> a);
    Tensor<T> TensorLog<T>(Tensor<T> a);
    Tensor<T> TensorSqrt<T>(Tensor<T> a);

    // NEW: Activations
    Tensor<T> ReLU<T>(Tensor<T> a);
    Tensor<T> Sigmoid<T>(Tensor<T> a);
    Tensor<T> Tanh<T>(Tensor<T> a);
    Tensor<T> Softmax<T>(Tensor<T> a, int axis);

    // NEW: Reductions
    Tensor<T> ReduceSum<T>(Tensor<T> a, int[] axes, bool keepDims);
    Tensor<T> ReduceMean<T>(Tensor<T> a, int[] axes, bool keepDims);
    Tensor<T> ReduceMax<T>(Tensor<T> a, int[] axes, bool keepDims);

    // NEW: Pooling
    Tensor<T> MaxPool2D<T>(Tensor<T> input, int[] kernelSize, int[] stride, int[] padding);
    Tensor<T> AvgPool2D<T>(Tensor<T> input, int[] kernelSize, int[] stride, int[] padding);

    // NEW: Convolution
    Tensor<T> Conv2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding);
    Tensor<T> ConvTranspose2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding);

    // NEW: Normalization
    Tensor<T> BatchNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, Tensor<T> mean, Tensor<T> variance, T epsilon);
    Tensor<T> LayerNorm<T>(Tensor<T> input, int[] normalizedShape, Tensor<T>? gamma, Tensor<T>? beta, T epsilon);

    // NEW: Shape operations
    Tensor<T> Reshape<T>(Tensor<T> a, int[] newShape);
    Tensor<T> Concat<T>(List<Tensor<T>> tensors, int axis);
    Tensor<T> Pad<T>(Tensor<T> a, int[,] padWidth, T value);
}
```

**Estimated Effort**: 12-20 hours
**Priority**: CRITICAL
**Files**:
- `src/Interfaces/IEngine.cs`
- `src/Engines/CpuEngine.cs`
- `src/Engines/GpuEngine.cs`

---

### US-ENGINE-3: Refactor All TensorOperations to Use IEngine
**As a** library maintainer
**I need** all TensorOperations to use IEngine consistently
**So that** CPU/GPU acceleration works automatically

**Dependencies**: US-ENGINE-2 complete

**Acceptance Criteria**:
- All 43+ TensorOperations methods refactored
- Replace direct Tensor methods with IEngine calls
- Replace scalar loops with IEngine vectorized operations
- Backward compatibility maintained
- All unit tests pass
- Performance tests show improvement

**Example Refactorings**:
```csharp
// BEFORE: Subtract (line 206)
public static ComputationNode<T> Subtract(ComputationNode<T> a, ComputationNode<T> b)
{
    var result = a.Value.ElementwiseSubtract(b.Value);  // ❌ Direct
    // ... backward function ...
}

// AFTER:
public static ComputationNode<T> Subtract(ComputationNode<T> a, ComputationNode<T> b)
{
    var engine = AiDotNetEngine.Current;
    var result = engine.TensorSubtract(a.Value, b.Value);  // ✅ IEngine
    // ... backward function ...
}

// BEFORE: ReLU (line 794)
public static ComputationNode<T> ReLU(ComputationNode<T> a)
{
    var numOps = MathHelper.GetNumericOperations<T>();
    var result = a.Value.Transform((x, _) =>
        numOps.Compare(x, numOps.Zero) > 0 ? x : numOps.Zero);  // ❌ Slow scalar loop
    // ... backward function ...
}

// AFTER:
public static ComputationNode<T> ReLU(ComputationNode<T> a)
{
    var engine = AiDotNetEngine.Current;
    var result = engine.ReLU(a.Value);  // ✅ Vectorized GPU-accelerated
    // ... backward function ...
}
```

**Estimated Effort**: 15-25 hours (40+ methods to refactor)
**Priority**: HIGH
**Files**: `src/Autodiff/TensorOperations.cs`

---

### US-ENGINE-4: Performance Benchmarks
**As a** performance engineer
**I need** comprehensive benchmarks
**So that** we validate CPU/GPU acceleration gains

**Acceptance Criteria**:
- Benchmark old (direct) vs new (IEngine) for all operations
- CPU vs GPU comparisons
- Small (10x10) vs large (1000x1000) tensors
- Performance report documenting speedups
- Regression tests

**Estimated Effort**: 6-10 hours
**Priority**: MEDIUM
**Files**: `tests/Performance/TensorOperationsEngineBenchmarks.cs` (new)

---

## Layer Implementations (65 Remaining)

**Note**: After US-ARCH-1 and US-ARCH-2, all layers MUST implement ExportComputationGraph

### US-LAYER-01: Basic/Structural Layers (6 remaining)
**Layers**:
- ⚠️ FlattenLayer
- ⚠️ ReshapeLayer
- ⚠️ DenseLayer
- ⚠️ FeedForwardLayer

**Already Complete** (4):
- ✅ ActivationLayer
- ✅ InputLayer
- ✅ DropoutLayer
- ✅ MultiplyLayer
- ✅ AddLayer
- ✅ FullyConnectedLayer

**Acceptance Criteria** (per layer):
- Implement ExportComputationGraph with symbolic inputs
- Set SupportsJitCompilation = true
- Unit tests
- Integration test with JitCompiler.Compile()

**Estimated Effort**: 2-3 hours (30-45 min per layer)
**Priority**: HIGH
**Files**: `src/NeuralNetworks/Layers/{Flatten,Reshape,Dense,FeedForward}Layer.cs`

---

### US-LAYER-02: Normalization Layers (2 layers)
**Layers**:
- ⚠️ BatchNormalizationLayer
- ⚠️ LayerNormalizationLayer

**Acceptance Criteria**:
- Export graph with gamma, beta, running mean/var
- Handle inference mode (no gradient tracking)
- Unit tests
- Integration tests

**Estimated Effort**: 3-5 hours (1.5-2.5 hours per layer)
**Priority**: HIGH
**Files**: `src/NeuralNetworks/Layers/{BatchNormalization,LayerNormalization}Layer.cs`

---

### US-LAYER-03: Pooling Layers (2 remaining)
**Layers**:
- ⚠️ AvgPoolingLayer
- ⚠️ PoolingLayer

**Already Complete** (2):
- ✅ MaxPoolingLayer
- ✅ GlobalPoolingLayer

**Acceptance Criteria**:
- Export pooling operation with kernel, stride, padding
- Unit tests with various configurations
- Integration tests

**Estimated Effort**: 2-3 hours (1-1.5 hours per layer)
**Priority**: MEDIUM
**Files**: `src/NeuralNetworks/Layers/{AvgPooling,Pooling}Layer.cs`

---

### US-LAYER-04: Convolutional Layers (7 layers)
**Layers**:
- ⚠️ ConvolutionalLayer
- ⚠️ DeconvolutionalLayer
- ⚠️ SeparableConvolutionalLayer
- ⚠️ DepthwiseSeparableConvolutionalLayer
- ⚠️ DilatedConvolutionalLayer
- ⚠️ SubpixelConvolutionalLayer
- ⚠️ LocallyConnectedLayer

**Acceptance Criteria**:
- Export conv operations with kernel, stride, padding, dilation
- Handle weight tensors correctly
- Unit tests
- Integration tests with CNNs

**Estimated Effort**: 8-12 hours (1-2 hours per layer)
**Priority**: HIGH
**Files**: `src/NeuralNetworks/Layers/{Convolutional,Deconvolutional,Separable*,Dilated*,Subpixel*,LocallyConnected}Layer.cs`

---

### US-LAYER-05: Attention Layers (4 layers)
**Layers**:
- ⚠️ AttentionLayer
- ⚠️ SelfAttentionLayer
- ⚠️ MultiHeadAttentionLayer
- ⚠️ SqueezeAndExcitationLayer

**Acceptance Criteria**:
- Export Q, K, V projections
- Handle scaled dot-product attention
- Multi-head splitting/concatenation
- Unit tests
- Integration tests with transformers

**Estimated Effort**: 5-8 hours (1.5-2 hours per layer)
**Priority**: HIGH
**Files**: `src/NeuralNetworks/Layers/{Attention,SelfAttention,MultiHeadAttention,SqueezeAndExcitation}Layer.cs`

---

### US-LAYER-06: Recurrent Layers (5 layers)
**Layers**:
- ⚠️ RecurrentLayer
- ⚠️ LSTMLayer
- ⚠️ GRULayer
- ⚠️ BidirectionalLayer
- ⚠️ ConvLSTMLayer

**Acceptance Criteria**:
- Export unrolled recurrent operations
- Handle hidden state initialization
- Bidirectional forward/backward passes
- Unit tests
- Integration tests with RNNs

**Estimated Effort**: 8-12 hours (1.5-2.5 hours per layer)
**Priority**: HIGH
**Files**: `src/NeuralNetworks/Layers/{Recurrent,LSTM,GRU,Bidirectional,ConvLSTM}Layer.cs`

---

### US-LAYER-07: Embedding Layers (3 layers)
**Layers**:
- ⚠️ EmbeddingLayer
- ⚠️ PositionalEncodingLayer
- ⚠️ PatchEmbeddingLayer

**Acceptance Criteria**:
- Export embedding lookups
- Positional encoding generation
- Patch extraction and embedding
- Unit tests
- Integration tests

**Estimated Effort**: 4-6 hours (1.5-2 hours per layer)
**Priority**: MEDIUM
**Files**: `src/NeuralNetworks/Layers/{Embedding,PositionalEncoding,PatchEmbedding}Layer.cs`

---

### US-LAYER-08: Transformer Layers (3 layers)
**Layers**:
- ⚠️ TransformerEncoderLayer
- ⚠️ TransformerDecoderLayer
- ⚠️ DecoderLayer

**Acceptance Criteria**:
- Export encoder/decoder blocks
- Multi-head attention + feed-forward
- Residual connections + layer norm
- Unit tests
- Integration tests with transformers

**Estimated Effort**: 6-10 hours (2-3 hours per layer)
**Priority**: HIGH
**Files**: `src/NeuralNetworks/Layers/{TransformerEncoder,TransformerDecoder,Decoder}Layer.cs`

---

### US-LAYER-09: Regularization/Noise (2 layers)
**Layers**:
- ⚠️ GaussianNoiseLayer
- ⚠️ MaskingLayer

**Acceptance Criteria**:
- Export noise addition (inference mode = no-op)
- Export masking operations
- Unit tests

**Estimated Effort**: 1-2 hours
**Priority**: LOW
**Files**: `src/NeuralNetworks/Layers/{GaussianNoise,Masking}Layer.cs`

---

### US-LAYER-10: Spatial/Shape Manipulation (4 remaining)
**Layers**:
- ⚠️ CroppingLayer
- ⚠️ UpsamplingLayer
- ⚠️ SpatialTransformerLayer
- ⚠️ SpatialPoolerLayer

**Already Complete** (1):
- ✅ PaddingLayer

**Acceptance Criteria**:
- Export cropping/upsampling operations
- Spatial transformation
- Unit tests

**Estimated Effort**: 3-5 hours
**Priority**: MEDIUM
**Files**: `src/NeuralNetworks/Layers/{Cropping,Upsampling,SpatialTransformer,SpatialPooler}Layer.cs`

---

### US-LAYER-11: Utility/Wrapper (3 remaining)
**Layers**:
- ⚠️ TimeDistributedLayer
- ⚠️ LambdaLayer

**Already Complete** (2):
- ✅ MeanLayer
- ✅ ConcatenateLayer
- ✅ SplitLayer

**Acceptance Criteria**:
- TimeDistributed: Apply inner layer to time steps
- Lambda: Custom operations
- Unit tests

**Estimated Effort**: 2-4 hours
**Priority**: MEDIUM
**Files**: `src/NeuralNetworks/Layers/{TimeDistributed,Lambda}Layer.cs`

---

### US-LAYER-12: Advanced Architectures (11 layers)
**Layers**:
- ⚠️ ResidualLayer
- ⚠️ HighwayLayer
- ⚠️ GatedLinearUnitLayer
- ⚠️ CapsuleLayer
- ⚠️ PrimaryCapsuleLayer
- ⚠️ DigitCapsuleLayer
- ⚠️ GraphConvolutionalLayer
- ⚠️ MixtureOfExpertsLayer
- ⚠️ ExpertLayer
- ⚠️ RBFLayer
- ⚠️ RBMLayer

**Acceptance Criteria**:
- Export residual/highway connections
- Capsule routing mechanisms
- Graph convolutions
- MoE gating
- Unit tests

**Estimated Effort**: 12-18 hours (1-2 hours per layer)
**Priority**: MEDIUM
**Files**: Multiple files in `src/NeuralNetworks/Layers/`

---

### US-LAYER-13: Memory/Attention Specialized (7 layers)
**Layers**:
- ⚠️ MemoryReadLayer
- ⚠️ MemoryWriteLayer
- ⚠️ TemporalMemoryLayer
- ⚠️ ContinuumMemorySystemLayer
- ⚠️ ReadoutLayer
- ⚠️ ReconstructionLayer
- ⚠️ RepParameterizationLayer

**Acceptance Criteria**:
- Export memory operations
- VAE reparameterization
- Unit tests

**Estimated Effort**: 8-12 hours (1-2 hours per layer)
**Priority**: LOW
**Files**: Multiple files in `src/NeuralNetworks/Layers/`

---

### US-LAYER-14: Biological/Neuromorphic (3 layers)
**Layers**:
- ⚠️ SpikingLayer
- ⚠️ SynapticPlasticityLayer
- ⚠️ ReservoirLayer

**Acceptance Criteria**:
- Export spiking dynamics (temporal approximation)
- Reservoir computing
- Unit tests

**Estimated Effort**: 4-8 hours (1.5-3 hours per layer)
**Priority**: DEFER
**Files**: Multiple files in `src/NeuralNetworks/Layers/`

---

### US-LAYER-15: Experimental/Special (4 layers)
**Layers**:
- ⚠️ QuantumLayer
- ⚠️ LogVarianceLayer
- ⚠️ MeasurementLayer
- ⚠️ AnomalyDetectorLayer
- ⚠️ ConditionalRandomFieldLayer

**Acceptance Criteria**:
- Export specialized operations
- Unit tests

**Estimated Effort**: 4-8 hours
**Priority**: DEFER
**Files**: Multiple files in `src/NeuralNetworks/Layers/`

---

## Model Implementations (115 Models)

### US-MODEL-01: Regression Models - Tier 1 (Simple Linear) (4 models)
**Models**:
1. SimpleRegression
2. MultipleRegression
3. PolynomialRegression
4. BayesianRegression

**Acceptance Criteria** (per model):
- Implement IFullModel (which includes IJitCompilable) if not already
- ExportComputationGraph returns complete inference graph
- SupportsJitCompilation = true
- Unit tests
- Integration tests showing 5-10x speedup
- Accuracy validation (JIT vs normal within 1e-6)

**Estimated Effort**: 8-12 hours (2-3 hours per model)
**Priority**: HIGH
**Files**: `src/Regression/{Simple,Multiple,Polynomial,Bayesian}Regression.cs`

---

### US-MODEL-02: Regression Models - Tier 2 (Regularized) (4 models)
**Models**:
5. LogisticRegression
6. MultinomialLogisticRegression
7. KernelRidgeRegression
8. SupportVectorRegression

**Estimated Effort**: 8-12 hours
**Priority**: HIGH
**Files**: `src/Regression/{Logistic,MultinomialLogistic,KernelRidge,SupportVector}Regression.cs`

---

### US-MODEL-03: Regression Models - Tier 3 (Tree-Based) (8 models)
**Models**:
9. DecisionTreeRegression
10. RandomForestRegression
11. GradientBoostingRegression
12. AdaBoostR2Regression
13. ExtremelyRandomizedTreesRegression
14. QuantileRegressionForests
15. M5ModelTreeRegression
16. ConditionalInferenceTreeRegression

**Challenge**: Decision trees need if-then-else in computation graphs

**Estimated Effort**: 16-24 hours (2-3 hours per model)
**Priority**: MEDIUM
**Files**: Multiple files in `src/Regression/`

---

### US-MODEL-04: Regression Models - Tier 4 (Neural) (2 models)
**Models**:
17. NeuralNetworkRegression
18. MultilayerPerceptronRegression

**Note**: Should delegate to neural network base classes

**Estimated Effort**: 4-6 hours
**Priority**: HIGH
**Files**: `src/Regression/{NeuralNetwork,MultilayerPerceptron}Regression.cs`

---

### US-MODEL-05: Regression Models - Tier 5 (Specialized) (20 models)
**Models**:
19. KNearestNeighborsRegression
20. LocallyWeightedRegression
21. IsotonicRegression
22. QuantileRegression
23. RobustRegression
24. MultivariateRegression
25. PoissonRegression
26. NegativeBinomialRegression
27. SplineRegression
28. GeneralizedAdditiveModelRegression
29. RadialBasisFunctionRegression
30. PartialLeastSquaresRegression
31. PrincipalComponentRegression
32. OrthogonalRegression
33. StepwiseRegression
34. WeightedRegression
35. SymbolicRegression
36. GeneticAlgorithmRegression
37. TimeSeriesRegression
38. GaussianProcessRegression

**Estimated Effort**: 30-50 hours (1.5-2.5 hours per model)
**Priority**: LOW
**Files**: Multiple files in `src/Regression/`

---

### US-MODEL-06: Time Series Models - Tier 1 (Classical) (6 models)
**Models**:
1. ARModel
2. MAModel
3. ARMAModel
4. ARIMAModel
5. SARIMAModel
6. VARMAModel

**Acceptance Criteria**:
- Export autoregressive operations
- Handle sequential dependencies
- Unit tests
- Forecasting accuracy validation

**Estimated Effort**: 10-16 hours (1.5-2.5 hours per model)
**Priority**: MEDIUM
**Files**: `src/TimeSeries/{AR,MA,ARMA,ARIMA,SARIMA,VARMA}Model.cs`

---

### US-MODEL-07: Time Series Models - Tier 2 (Advanced) (8 models)
**Models**:
7. ARIMAXModel
8. VectorAutoRegressionModel
9. ExponentialSmoothingModel
10. StateSpaceModel
11. UnobservedComponentsModel
12. GARCHModel
13. TBATSModel
14. InterventionAnalysisModel

**Estimated Effort**: 14-22 hours
**Priority**: MEDIUM
**Files**: Multiple files in `src/TimeSeries/`

---

### US-MODEL-08: Time Series Models - Tier 3 (Neural/Special) (6 models)
**Models**:
15. NBEATSModel
16. NeuralNetworkARIMAModel
17. ProphetModel
18. BayesianStructuralTimeSeriesModel
19. SpectralAnalysisModel
20. TransferFunctionModel

**Estimated Effort**: 10-16 hours
**Priority**: LOW
**Files**: Multiple files in `src/TimeSeries/`

---

### US-MODEL-09: Neural Network Models - Tier 1 (Core) (9 models)
**Models**:
1. NeuralNetwork
2. FeedForwardNeuralNetwork
3. ConvolutionalNeuralNetwork
4. RecurrentNeuralNetwork
5. LSTMNeuralNetwork
6. GRUNeuralNetwork
7. ResidualNeuralNetwork
8. Autoencoder
9. VariationalAutoencoder

**Key**: Most inherit from NeuralNetworkBase, so they get JIT for FREE after US-BASE-1!

**Acceptance Criteria** (per model):
- Verify inherits from NeuralNetworkBase
- Override ExportComputationGraph() only if custom architecture
- Unit tests
- Integration tests

**Estimated Effort**: 10-16 hours (simple ones inherit, complex ones need overrides)
**Priority**: CRITICAL
**Files**: `src/NeuralNetworks/{NeuralNetwork,FeedForward*,Convolutional*,Recurrent*,LSTM*,GRU*,Residual*,Autoencoder,VariationalAutoencoder}.cs`

---

### US-MODEL-10: Neural Network Models - Tier 2 (Transformers) (4 models)
**Models**:
10. Transformer
11. VisionTransformer
12. AttentionNetwork
13. SuperNet

**Estimated Effort**: 8-12 hours
**Priority**: HIGH
**Files**: `src/NeuralNetworks/{Transformer,VisionTransformer,AttentionNetwork,SuperNet}.cs`

---

### US-MODEL-11: Neural Network Models - Tier 3 (Generative) (3 models)
**Models**:
14. GenerativeAdversarialNetwork
15. RestrictedBoltzmannMachine
16. DeepBoltzmannMachine

**Challenge**: GAN has dual networks (generator + discriminator)

**Estimated Effort**: 6-10 hours
**Priority**: MEDIUM
**Files**: `src/NeuralNetworks/{GenerativeAdversarial*,RestrictedBoltzmann*,DeepBoltzmann*}.cs`

---

### US-MODEL-12: Neural Network Models - Tier 4 (Advanced Architectures) (8 models)
**Models**:
17. CapsuleNetwork
18. GraphNeuralNetwork
19. MixtureOfExpertsNeuralNetwork
20. SiameseNetwork
21. RadialBasisFunctionNetwork
22. SelfOrganizingMap
23. ExtremeLearningMachine
24. DeepBeliefNetwork

**Estimated Effort**: 14-22 hours
**Priority**: MEDIUM
**Files**: Multiple files in `src/NeuralNetworks/`

---

### US-MODEL-13: Neural Network Models - Tier 5 (Memory/Special) (6 models)
**Models**:
25. NeuralTuringMachine
26. DifferentiableNeuralComputer
27. MemoryNetwork
28. HopfieldNetwork
29. HopeNetwork
30. EchoStateNetwork

**Estimated Effort**: 10-16 hours
**Priority**: LOW
**Files**: Multiple files in `src/NeuralNetworks/`

---

### US-MODEL-14: Neural Network Models - Tier 6 (Experimental) (8 models)
**Models**:
31. SpikingNeuralNetwork
32. QuantumNeuralNetwork
33. LiquidStateMachine
34. DeepQNetwork
35. HTMNetwork
36. OccupancyNeuralNetwork
37. NEAT

**Estimated Effort**: 14-24 hours
**Priority**: DEFER
**Files**: Multiple files in `src/NeuralNetworks/`

---

### US-MODEL-15: Ensemble Models (10 models)
**Models**:
1. AdaptiveTeacherModel
2. CurriculumTeacherModel
3. DistributedTeacherModel
4. EnsembleTeacherModel
5. MultiModalTeacherModel
6. OnlineTeacherModel
7. PretrainedTeacherModel
8. QuantizedTeacherModel
9. SelfTeacherModel
10. TransformerTeacherModel

**Estimated Effort**: 15-25 hours
**Priority**: LOW
**Files**: `src/KnowledgeDistillation/Teachers/*.cs`

---

### US-MODEL-16: Distributed Models (9 models)
**Models**:
1. DDPModel
2. FSDPModel
3. HybridShardedModel
4. PipelineParallelModel
5. TensorParallelModel
6. ZeRO1Model
7. ZeRO2Model
8. ZeRO3Model
9. PartitionedModel

**Note**: These are wrappers, should delegate to underlying model

**Estimated Effort**: 8-15 hours
**Priority**: LOW
**Files**: `src/DistributedTraining/*.cs`, `src/Deployment/Edge/*.cs`

---

## Integration & Testing

### US-TEST-1: Priority 1 Integration (from original gap analysis)
**As a** developer
**I need** end-to-end JIT working with one model
**So that** the feature is minimally viable

**Work** (from original analysis):
- US-1.1: TensorOperations metadata (3-5 hours)
- US-1.2: Reference model (LinearRegression) (8-12 hours)
- US-1.3: PredictionModelBuilder integration (5-8 hours)
- US-1.4: PredictionModelResult integration (3-5 hours)
- US-1.5: Integration tests (8-12 hours)

**Estimated Effort**: 27-42 hours
**Priority**: CRITICAL

---

## Estimation Summary

### Critical Path (Must Be Done In Order)

**Phase 0: Fix Architecture** (10-19 hours) - BLOCKING EVERYTHING
1. US-ARCH-1: LayerBase abstract methods (1-2 hours)
2. US-ARCH-2: Remove NeuralNetworkBase violations (4-8 hours)
3. US-BASE-1: Implement NeuralNetworkBase JIT (2-4 hours)
4. US-ARCH-3: Model audit (3-5 hours)

**Phase 1: Core Infrastructure** (69-128 hours)
5. US-ENGINE-1,2,3,4: TensorOperations IEngine (37-61 hours)
6. US-LAYER-01 through 15: All layers (32-48 hours estimated total across 65 layers)

**Phase 2: Priority 1 Integration** (27-42 hours)
7. US-TEST-1: Complete Priority 1 blockers from original gap analysis

**Phase 3: Model Implementations** (~150-300 hours)
8. US-MODEL-01 through 16: All models by tier/priority

### Total Estimate: 256-489 hours (6-12 weeks with 1 developer)

### Parallelization (4 developers):
- Developer A: Phase 0 + IEngine work (47-80 hours)
- Developer B: Layers batch 1 (core layers: 20-35 hours)
- Developer C: Layers batch 2 (advanced layers: 20-35 hours)
- Developer D: Priority 1 integration + models (30-50 hours)

**Time to production-ready**: 3-4 weeks with 4 developers

---

## Next Steps

1. Get user approval for Phase 0 approach
2. Execute US-ARCH-1, US-ARCH-2, US-BASE-1, US-ARCH-3 in order
3. Complete US-ARCH-3 model audit before planning detailed model work
4. Parallel execution of Phase 1 work
5. Continuous integration and testing throughout

---

## Questions for User

1. Approve Phase 0 critical path and order of execution?
2. Should I start US-ARCH-1 immediately (LayerBase abstract methods)?
3. Which model tiers are highest priority after neural networks?
4. Should experimental models (Quantum, Spiking, NEAT) be explicitly deferred?
5. Target timeline for complete work?
6. Separate PRs for each phase or one large PR?

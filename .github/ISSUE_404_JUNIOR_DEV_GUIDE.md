# Issue #404: Junior Developer Implementation Guide
## Neural Architecture Search (NAS) - DARTS and ENAS

---

## Table of Contents
1. [Understanding Neural Architecture Search](#understanding-neural-architecture-search)
2. [DARTS: Differentiable Architecture Search](#darts-differentiable-architecture-search)
3. [ENAS: Efficient Neural Architecture Search](#enas-efficient-neural-architecture-search)
4. [Search Spaces and Operations](#search-spaces-and-operations)
5. [Implementation Strategy](#implementation-strategy)
6. [Testing Strategy](#testing-strategy)
7. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)

---

## Understanding Neural Architecture Search

### What is Neural Architecture Search?

Neural Architecture Search (NAS) is the process of **automatically designing neural network architectures** instead of manually engineering them.

**Traditional Approach**:
```
Human Expert → Design Architecture → Train → Evaluate → Iterate
                     ↑ Requires deep expertise and intuition
```

**NAS Approach**:
```
Search Algorithm → Generate Architecture → Train → Evaluate → Update Search
                         ↑ Automated discovery of optimal architectures
```

### Why NAS?

1. **Removes Human Bias**: Discovers architectures humans might not imagine
2. **Task-Specific Optimization**: Finds architectures tailored to specific datasets/tasks
3. **Saves Time**: Automates the trial-and-error process
4. **Better Performance**: Often outperforms hand-designed architectures

### The NAS Problem

**Goal**: Find the best architecture `A*` from search space `A`:

```
A* = argmax_{A ∈ A} Accuracy(A, D_val)
subject to: Latency(A) < T_max
            Params(A) < P_max
```

Where:
- `A` = Set of possible architectures
- `D_val` = Validation dataset
- `T_max` = Maximum inference latency constraint
- `P_max` = Maximum parameter count constraint

### NAS Components

1. **Search Space**: Set of possible architectures
2. **Search Strategy**: Algorithm to explore the search space
3. **Performance Estimation**: How to evaluate architectures efficiently

---

## DARTS: Differentiable Architecture Search

### Core Idea

Instead of searching discrete architectures, **relax the search space to be continuous** and use gradient descent to optimize architecture parameters.

### Key Innovation

**Discrete Search** (Slow):
```
Try architecture A1 → Evaluate → Try A2 → Evaluate → ...
```

**Continuous Relaxation** (Fast):
```
Mix all operations → Use gradients to find optimal mixture weights
```

### Mathematical Foundation

#### 1. Continuous Relaxation

Instead of choosing one operation, use a **weighted sum of all operations**:

```
Traditional (discrete):
o^(i,j)(x) = operation_k(x)  where k is selected from {conv3x3, conv5x5, maxpool, ...}

DARTS (continuous):
o^(i,j)(x) = Σ_k α_k * operation_k(x)
```

Where:
- `α_k` = Importance weight for operation k (architecture parameters)
- Operations are mixed with softmax: `α_k = exp(α_k) / Σ_j exp(α_j)`

#### 2. Bilevel Optimization

DARTS optimizes two sets of parameters simultaneously:

```
min_α L_val(w*(α), α)          [Architecture optimization]
s.t. w*(α) = argmin_w L_train(w, α)  [Weight optimization]
```

Where:
- `α` = Architecture parameters (which operations to use)
- `w` = Network weights (learned via backprop)
- `L_train` = Training loss
- `L_val` = Validation loss

**Intuition**:
- Train weights `w` to minimize training loss (given architecture `α`)
- Update architecture `α` to minimize validation loss (given weights `w`)

#### 3. Gradient-Based Search

Both `w` and `α` are optimized using gradient descent:

```
// Step 1: Update weights w (standard training)
w ← w - ξ * ∇_w L_train(w, α)

// Step 2: Update architecture α (architecture search)
α ← α - η * ∇_α L_val(w, α)
```

#### 4. Approximation of ∇_α

Computing the exact gradient `∇_α L_val(w*(α), α)` requires computing `w*(α)` (expensive).

**Approximation** (first-order):
```
∇_α L_val(w, α) ≈ ∇_α L_val(w - ξ * ∇_w L_train(w, α), α)
```

**Algorithm**:
```
1. Compute ∇_w L_train(w, α)
2. Create temporary weights: w' = w - ξ * ∇_w L_train(w, α)
3. Compute ∇_α L_val(w', α)  [gradient of architecture params]
4. Update α using this gradient
```

### DARTS Search Space

**Cell-Based Search**:
- Define a **cell** (a small subgraph)
- Stack multiple cells to form the full network
- Search for optimal cell structure

**Node Representation**:
```
Cell = {node_0, node_1, ..., node_N}

node_j = Σ_{i<j} Σ_o softmax(α_{i,j})_o * operation_o(node_i)
```

Where:
- Each node is a feature map
- `node_0, node_1` = Input nodes (from previous cells)
- `node_j` = Intermediate node (for j >= 2)
- `operation_o` = One of {conv3x3, conv5x5, dilated_conv, sep_conv, max_pool, avg_pool, skip, zero}

**Example Cell**:
```
Input 0 ──┐
          ├→ [Mixed Op] → Node 2 ─┐
Input 1 ──┤                       ├→ [Mixed Op] → Node 3 → Output
          └→ [Mixed Op] → Node 2 ─┘
```

### DARTS Algorithm

```
Algorithm: DARTS
Input: Training data D_train, validation data D_val
Output: Best architecture α*

1. Initialize:
   - Architecture parameters α randomly
   - Network weights w randomly

2. For each search epoch t = 1..T:
   a. Sample mini-batch from D_train
   b. Update w: w ← w - ξ * ∇_w L_train(w, α)

   c. Sample mini-batch from D_val
   d. Compute w' = w - ξ * ∇_w L_train(w, α)  [virtual step]
   e. Update α: α ← α - η * ∇_α L_val(w', α)

3. Discretization:
   For each edge (i, j):
      Select operation with highest α: o* = argmax_o α_{i,j,o}

4. Return discretized architecture α*
```

### Implementation Sketch

```csharp
public class DARTSSearcher<T> where T : struct
{
    private Tensor<T> _architectureParams;  // α parameters
    private NeuralNetwork<T> _searchNetwork;  // Mixed network
    private IOptimizer<T> _weightOptimizer;
    private IOptimizer<T> _archOptimizer;

    public void Search(int numEpochs)
    {
        for (int epoch = 0; epoch < numEpochs; epoch++)
        {
            // Phase 1: Update network weights w
            var trainBatch = SampleBatch(_trainData);
            var trainLoss = _searchNetwork.Forward(trainBatch.X, trainBatch.Y);
            var wGradients = _searchNetwork.Backward();
            _weightOptimizer.Update(_searchNetwork.Weights, wGradients);

            // Phase 2: Update architecture parameters α
            // 2a. Compute virtual weights w'
            var wPrime = ComputeVirtualWeights();

            // 2b. Evaluate on validation data
            var valBatch = SampleBatch(_valData);
            _searchNetwork.SetWeights(wPrime);  // Temporarily use w'
            var valLoss = _searchNetwork.Forward(valBatch.X, valBatch.Y);

            // 2c. Compute ∇_α L_val(w', α)
            var alphaGradients = ComputeArchitectureGradients(valLoss);
            _archOptimizer.Update(_architectureParams, alphaGradients);

            // Restore original weights
            _searchNetwork.SetWeights(_originalWeights);

            // Log progress
            if (epoch % 10 == 0)
                LogTopOperations();
        }

        // Discretization: select best operations
        return DiscretizeArchitecture();
    }

    private Tensor<T> ComputeVirtualWeights()
    {
        // w' = w - ξ * ∇_w L_train(w, α)
        var gradients = _searchNetwork.ComputeGradients(_trainData);
        var virtualWeights = _searchNetwork.Weights.Clone();

        for (int i = 0; i < virtualWeights.Length; i++)
        {
            var step = NumOps<T>.Multiply(
                gradients[i],
                NumOps<T>.FromDouble(_weightOptimizer.LearningRate)
            );
            virtualWeights[i] = NumOps<T>.Subtract(virtualWeights[i], step);
        }

        return virtualWeights;
    }
}
```

---

## ENAS: Efficient Neural Architecture Search

### Core Idea

**Observation**: Training each architecture from scratch is wasteful.

**Solution**: Use **parameter sharing** across architectures.
- All architectures share the same set of weights
- Searching becomes selecting which subgraph to use
- Train weights once, search over subgraphs

### Controller-Based Search

ENAS uses a **controller RNN** to generate architectures:

```
Controller RNN → Generates architecture A → Evaluate A → Update controller
                            ↑ Shares weights with all other architectures
```

### Mathematical Foundation

#### 1. Search Space as DAG

Represent architectures as Directed Acyclic Graphs (DAGs):

```
Nodes: {0, 1, 2, ..., N}
Edges: (i, j) with operation o_{i,j}
```

#### 2. Controller Policy

The controller samples architectures from a policy `π(A; θ)`:

```
θ = Controller parameters (LSTM weights)
A = Architecture sampled from π(θ)
```

**Architecture Encoding**:
```
For each node j = 2..N:
    1. Sample previous node: i ~ Categorical(π_i)
    2. Sample operation: o ~ Categorical(π_o)

Architecture A = {(i, j, o) for all j}
```

#### 3. Reward Signal

Train controller to maximize expected reward:

```
J(θ) = E_{A~π(θ)}[R(A)]
```

Where:
- `R(A)` = Accuracy of architecture A on validation set
- Maximize via REINFORCE algorithm

#### 4. REINFORCE Update

```
∇_θ J(θ) = E_{A~π(θ)}[R(A) * ∇_θ log π(A; θ)]

Approximated by sampling:
∇_θ J(θ) ≈ (1/M) * Σ_m R(A_m) * ∇_θ log π(A_m; θ)
```

**With Baseline** (reduce variance):
```
∇_θ J(θ) ≈ (1/M) * Σ_m (R(A_m) - b) * ∇_θ log π(A_m; θ)

where b = exponential moving average of past rewards
```

### ENAS Algorithm

```
Algorithm: ENAS
Input: Training data D_train, validation data D_val
Output: Best architecture A*

1. Initialize:
   - Shared weights w randomly
   - Controller parameters θ randomly
   - Baseline b = 0

2. For each search epoch t = 1..T:

   a. Train shared weights:
      For k = 1..K:
         - Sample architecture A from π(θ)
         - Sample mini-batch from D_train
         - Compute gradients ∇_w L(A, w) for architecture A
         - Update w using gradients

   b. Train controller:
      For m = 1..M:
         - Sample architecture A_m from π(θ)
         - Evaluate A_m on D_val to get reward R_m
         - Compute policy gradient:
           ∇_θ J ≈ (R_m - b) * ∇_θ log π(A_m; θ)
         - Update θ
         - Update baseline: b ← 0.95 * b + 0.05 * R_m

3. Return architecture with highest reward: A* = argmax_A R(A)
```

### Implementation Sketch

```csharp
public class ENASSearcher<T> where T : struct
{
    private NeuralNetwork<T> _sharedWeights;  // Shared across all architectures
    private LSTMController<T> _controller;     // Generates architectures
    private double _baseline;                  // For variance reduction

    public void Search(int numEpochs)
    {
        for (int epoch = 0; epoch < numEpochs; epoch++)
        {
            // Phase 1: Train shared weights
            TrainSharedWeights(numIterations: 50);

            // Phase 2: Train controller
            TrainController(numSamples: 10);

            if (epoch % 10 == 0)
                EvaluateTopArchitectures();
        }

        return FindBestArchitecture();
    }

    private void TrainSharedWeights(int numIterations)
    {
        for (int i = 0; i < numIterations; i++)
        {
            // Sample random architecture from controller
            var arch = _controller.SampleArchitecture();

            // Build network using sampled architecture
            var network = BuildNetwork(arch, _sharedWeights);

            // Train on mini-batch
            var batch = SampleBatch(_trainData);
            var loss = network.Forward(batch.X, batch.Y);
            var gradients = network.Backward();

            // Update ONLY the weights used in this architecture
            UpdateActiveWeights(arch, gradients);
        }
    }

    private void TrainController(int numSamples)
    {
        var rewards = new List<double>();
        var logProbs = new List<Tensor<T>>();

        // Sample multiple architectures
        for (int i = 0; i < numSamples; i++)
        {
            var (arch, logProb) = _controller.SampleWithLogProb();

            // Evaluate architecture
            var network = BuildNetwork(arch, _sharedWeights);
            var accuracy = EvaluateOnValidation(network);
            rewards.Add(accuracy);
            logProbs.Add(logProb);
        }

        // Compute policy gradient with baseline
        for (int i = 0; i < numSamples; i++)
        {
            var advantage = rewards[i] - _baseline;
            var gradient = logProbs[i].Multiply(NumOps<T>.FromDouble(advantage));
            _controller.ApplyGradient(gradient);
        }

        // Update baseline (exponential moving average)
        _baseline = 0.95 * _baseline + 0.05 * rewards.Average();
    }
}
```

### Controller Architecture

```csharp
public class LSTMController<T> where T : struct
{
    private LSTM<T> _lstm;
    private DenseLayer<T> _nodeSelector;     // Selects which previous node
    private DenseLayer<T> _operationSelector;  // Selects which operation

    public (Architecture, Tensor<T>) SampleWithLogProb()
    {
        var architecture = new Architecture();
        var totalLogProb = 0.0;
        var hiddenState = _lstm.InitialState();

        // For each node in the cell
        for (int nodeIdx = 2; nodeIdx < NumNodes; nodeIdx++)
        {
            // Step 1: Select previous node index
            var nodeLogits = _nodeSelector.Forward(hiddenState);
            var nodeProbs = Softmax(nodeLogits);
            var selectedNode = SampleCategorical(nodeProbs);
            totalLogProb += Math.Log(nodeProbs[selectedNode]);

            // Step 2: Select operation type
            var opLogits = _operationSelector.Forward(hiddenState);
            var opProbs = Softmax(opLogits);
            var selectedOp = SampleCategorical(opProbs);
            totalLogProb += Math.Log(opProbs[selectedOp]);

            // Add to architecture
            architecture.AddEdge(selectedNode, nodeIdx, (OperationType)selectedOp);

            // Update LSTM state
            var input = CreateEmbedding(selectedNode, selectedOp);
            hiddenState = _lstm.Step(input, hiddenState);
        }

        return (architecture, NumOps<T>.FromDouble(totalLogProb));
    }
}
```

---

## Search Spaces and Operations

### Primitive Operations

Common operations used in NAS search spaces:

```csharp
public enum OperationType
{
    // Convolutions
    Conv3x3,           // 3x3 convolution
    Conv5x5,           // 5x5 convolution
    Conv1x1,           // 1x1 convolution (bottleneck)

    // Separable Convolutions (efficient)
    SepConv3x3,        // 3x3 separable convolution
    SepConv5x5,        // 5x5 separable convolution

    // Dilated Convolutions (larger receptive field)
    DilConv3x3,        // 3x3 dilated convolution
    DilConv5x5,        // 5x5 dilated convolution

    // Pooling
    MaxPool3x3,        // 3x3 max pooling
    AvgPool3x3,        // 3x3 average pooling

    // Identity and Zero
    Identity,          // Skip connection (no operation)
    Zero               // No connection
}
```

### Operation Implementations

```csharp
public class OperationFactory<T> where T : struct
{
    public IOperation<T> CreateOperation(OperationType type, int channels, int stride)
    {
        return type switch
        {
            OperationType.Conv3x3 => new Conv2DLayer<T>(channels, channels, 3, stride, padding: 1),
            OperationType.Conv5x5 => new Conv2DLayer<T>(channels, channels, 5, stride, padding: 2),
            OperationType.SepConv3x3 => CreateSeparableConv(channels, 3, stride),
            OperationType.DilConv3x3 => CreateDilatedConv(channels, 3, stride, dilation: 2),
            OperationType.MaxPool3x3 => new MaxPooling2D<T>(3, stride, padding: 1),
            OperationType.AvgPool3x3 => new AveragePooling2D<T>(3, stride, padding: 1),
            OperationType.Identity => new IdentityLayer<T>(),
            OperationType.Zero => new ZeroLayer<T>(),
            _ => throw new ArgumentException($"Unknown operation type: {type}")
        };
    }

    private IOperation<T> CreateSeparableConv(int channels, int kernel, int stride)
    {
        // Depthwise + Pointwise convolution
        return new SequentialLayer<T>(
            new DepthwiseConv2D<T>(channels, kernel, stride, padding: kernel / 2),
            new Conv2DLayer<T>(channels, channels, kernelSize: 1, stride: 1)
        );
    }

    private IOperation<T> CreateDilatedConv(int channels, int kernel, int stride, int dilation)
    {
        return new DilatedConv2D<T>(channels, channels, kernel, stride, dilation);
    }
}
```

### Mixed Operation (DARTS)

```csharp
public class MixedOperation<T> : IOperation<T> where T : struct
{
    private readonly List<IOperation<T>> _operations;
    private Tensor<T> _alphas;  // Softmax weights

    public MixedOperation(List<IOperation<T>> operations, Tensor<T> alphas)
    {
        _operations = operations;
        _alphas = alphas;
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        // Compute softmax of architecture parameters
        var weights = Softmax(_alphas);

        // Weighted sum of all operations
        Tensor<T>? output = null;

        for (int i = 0; i < _operations.Count; i++)
        {
            var opOutput = _operations[i].Forward(input);
            var weighted = opOutput.Multiply(weights[i]);

            output = output == null ? weighted : output.Add(weighted);
        }

        return output!;
    }

    public Tensor<T> Backward(Tensor<T> gradOutput)
    {
        // Backpropagate through all operations
        var weights = Softmax(_alphas);
        Tensor<T>? gradInput = null;

        for (int i = 0; i < _operations.Count; i++)
        {
            var opGrad = _operations[i].Backward(gradOutput.Multiply(weights[i]));
            gradInput = gradInput == null ? opGrad : gradInput.Add(opGrad);
        }

        // Also compute gradient w.r.t. alphas
        ComputeAlphaGradients(gradOutput);

        return gradInput!;
    }
}
```

### Cell-Based Search Space

```csharp
public class SearchCell<T> : ILayer<T> where T : struct
{
    private readonly int _numNodes;
    private readonly List<MixedOperation<T>> _mixedOps;
    private Tensor<T> _architectureParams;

    public SearchCell(int numNodes, int channels)
    {
        _numNodes = numNodes;
        _mixedOps = new List<MixedOperation<T>>();

        // Create mixed operations for all possible edges
        int numEdges = numNodes * (numNodes - 1) / 2;
        _architectureParams = InitializeArchitectureParams(numEdges, NumOperations);

        // Build mixed operations
        for (int j = 0; j < numNodes; j++)
        {
            for (int i = 0; i < j; i++)
            {
                var operations = CreateAllOperations(channels);
                var alphas = GetAlphasForEdge(i, j);
                _mixedOps.Add(new MixedOperation<T>(operations, alphas));
            }
        }
    }

    public Tensor<T> Forward(Tensor<T> input0, Tensor<T> input1)
    {
        // Nodes: [input0, input1, node2, node3, ..., nodeN]
        var nodes = new List<Tensor<T>> { input0, input1 };

        int edgeIdx = 0;
        for (int j = 2; j < _numNodes; j++)
        {
            Tensor<T>? nodeOutput = null;

            // Aggregate from all previous nodes
            for (int i = 0; i < j; i++)
            {
                var edgeOutput = _mixedOps[edgeIdx++].Forward(nodes[i]);
                nodeOutput = nodeOutput == null ? edgeOutput : nodeOutput.Add(edgeOutput);
            }

            nodes.Add(nodeOutput!);
        }

        // Concatenate all intermediate nodes (skip inputs)
        return ConcatenateNodes(nodes.Skip(2).ToList());
    }

    public Architecture Discretize()
    {
        // Select top-k operations for each node
        var architecture = new Architecture();
        int edgeIdx = 0;

        for (int j = 2; j < _numNodes; j++)
        {
            var topEdges = new List<(int node, int opIdx, double weight)>();

            for (int i = 0; i < j; i++)
            {
                var alphas = GetAlphasForEdge(i, j);
                var maxIdx = GetMaxIndex(alphas);
                var maxWeight = alphas[maxIdx];
                topEdges.Add((i, maxIdx, Convert.ToDouble(maxWeight)));
                edgeIdx++;
            }

            // Keep top-2 edges for each node
            var selectedEdges = topEdges.OrderByDescending(e => e.weight).Take(2);

            foreach (var (node, opIdx, _) in selectedEdges)
            {
                architecture.AddEdge(node, j, (OperationType)opIdx);
            }
        }

        return architecture;
    }
}
```

---

## Implementation Strategy

### Phase 1: Core Abstractions

#### Architecture Representation

```csharp
public class Architecture
{
    public List<ArchitectureEdge> Edges { get; set; } = new List<ArchitectureEdge>();
    public int NumNodes { get; set; }

    public void AddEdge(int fromNode, int toNode, OperationType operation)
    {
        Edges.Add(new ArchitectureEdge
        {
            FromNode = fromNode,
            ToNode = toNode,
            Operation = operation
        });
    }

    public NeuralNetwork<T> Build<T>(int inputChannels, int numClasses) where T : struct
    {
        // Convert architecture to executable network
        var layers = new List<ILayer<T>>();

        // ... build network from edges ...

        return new NeuralNetwork<T>(layers);
    }
}

public class ArchitectureEdge
{
    public int FromNode { get; set; }
    public int ToNode { get; set; }
    public OperationType Operation { get; set; }
}
```

#### Search Strategy Interface

```csharp
public interface IArchitectureSearchStrategy<T> where T : struct
{
    /// <summary>
    /// Execute architecture search and return best architecture.
    /// </summary>
    Architecture Search(SearchConfig config);

    /// <summary>
    /// Get current search progress.
    /// </summary>
    SearchProgress GetProgress();
}

public class SearchConfig
{
    public int SearchEpochs { get; set; } = 50;
    public int TrainEpochs { get; set; } = 600;
    public double LearningRate { get; set; } = 0.025;
    public double ArchitectureLearningRate { get; set; } = 3e-4;
    public int BatchSize { get; set; } = 64;
    public SearchSpaceType SearchSpace { get; set; } = SearchSpaceType.DARTS;
}
```

---

## Testing Strategy

### Unit Tests

```csharp
[TestClass]
public class DARTSTests
{
    [TestMethod]
    public void MixedOperation_ForwardPass_AggregatesCorrectly()
    {
        // Arrange: Create mixed operation with 3 primitive ops
        var ops = new List<IOperation<double>>
        {
            new Conv2DLayer<double>(16, 16, 3, 1, padding: 1),
            new MaxPooling2D<double>(3, 1, padding: 1),
            new IdentityLayer<double>()
        };

        var alphas = new Tensor<double>(new[] { 0.5, 0.3, 0.2 });  // Softmax weights
        var mixedOp = new MixedOperation<double>(ops, alphas);

        var input = CreateRandomTensor(1, 16, 32, 32);

        // Act
        var output = mixedOp.Forward(input);

        // Assert: Output shape matches input
        Assert.AreEqual(input.Shape, output.Shape);
    }

    [TestMethod]
    public void SearchCell_Discretization_SelectsTopOperations()
    {
        // Arrange: Create search cell with known alpha values
        var cell = new SearchCell<double>(numNodes: 4, channels: 16);

        // Manually set alphas to prefer certain operations
        cell.SetAlpha(edge: 0, operation: OperationType.Conv3x3, value: 5.0);
        cell.SetAlpha(edge: 0, operation: OperationType.MaxPool3x3, value: 0.1);

        // Act: Discretize
        var architecture = cell.Discretize();

        // Assert: Selected operation is Conv3x3
        Assert.IsTrue(architecture.ContainsOperation(OperationType.Conv3x3));
    }

    [TestMethod]
    public void DARTS_Search_ReducesValidationLoss()
    {
        // Arrange
        var searcher = new DARTSSearcher<double>();
        var (trainX, trainY, valX, valY) = CreateSmallDataset();

        // Act: Run search for 10 epochs
        var config = new SearchConfig { SearchEpochs = 10 };
        var initialLoss = EvaluateLoss(searcher.CurrentArchitecture, valX, valY);

        searcher.Search(config);

        var finalLoss = EvaluateLoss(searcher.BestArchitecture, valX, valY);

        // Assert: Validation loss decreased
        Assert.IsTrue(finalLoss < initialLoss);
    }
}
```

### Integration Tests

```csharp
[TestClass]
public class NASIntegrationTests
{
    [TestMethod]
    public void DARTS_CIFAR10_FindsCompetitiveArchitecture()
    {
        // Arrange: Load CIFAR-10
        var (trainX, trainY, testX, testY) = LoadCIFAR10();

        var searcher = new DARTSSearcher<double>();

        // Act: Search for 50 epochs
        var config = new SearchConfig
        {
            SearchEpochs = 50,
            LearningRate = 0.025,
            ArchitectureLearningRate = 3e-4
        };

        var architecture = searcher.Search(config);

        // Train discovered architecture from scratch
        var network = architecture.Build<double>(inputChannels: 3, numClasses: 10);
        network.Train(trainX, trainY, epochs: 600);

        // Assert: Achieves > 90% accuracy
        var accuracy = network.Evaluate(testX, testY);
        Assert.IsTrue(accuracy > 0.90, $"Expected > 90%, got {accuracy:F2}%");
    }

    [TestMethod]
    public void ENAS_Search_SharesWeightsCorrectly()
    {
        // Arrange
        var searcher = new ENASSearcher<double>();
        var (trainX, trainY, valX, valY) = LoadCIFAR10();

        // Act: Search
        var architecture = searcher.Search(new SearchConfig { SearchEpochs = 50 });

        // Assert: Shared weights were reused
        var totalTrainingTime = searcher.GetTotalTrainingTime();
        var expectedTime = 50 * AverageBatchTime;  // Should be ~50x, not 1000x

        Assert.IsTrue(totalTrainingTime < expectedTime * 2,
            "ENAS should be much faster than training each architecture separately");
    }
}
```

---

## Step-by-Step Implementation Guide

### Phase 1: Primitive Operations (Week 1)

**Step 1: Define Operation Interface**

File: `src/NAS/IOperation.cs`

```csharp
namespace AiDotNet.NAS
{
    public interface IOperation<T> where T : struct
    {
        Tensor<T> Forward(Tensor<T> input);
        Tensor<T> Backward(Tensor<T> gradOutput);
        int GetOutputChannels();
        (int height, int width) GetOutputShape(int inputHeight, int inputWidth);
    }
}
```

**Step 2: Implement Primitive Operations**

File: `src/NAS/Operations/PrimitiveOperations.cs`

```csharp
namespace AiDotNet.NAS.Operations
{
    public class SeparableConv2D<T> : IOperation<T> where T : struct
    {
        private readonly DepthwiseConv2D<T> _depthwise;
        private readonly Conv2DLayer<T> _pointwise;

        public SeparableConv2D(int channels, int kernelSize, int stride, int padding)
        {
            _depthwise = new DepthwiseConv2D<T>(channels, kernelSize, stride, padding);
            _pointwise = new Conv2DLayer<T>(channels, channels, kernelSize: 1, stride: 1);
        }

        public Tensor<T> Forward(Tensor<T> input)
        {
            var depthwiseOut = _depthwise.Forward(input);
            return _pointwise.Forward(depthwiseOut);
        }

        public Tensor<T> Backward(Tensor<T> gradOutput)
        {
            var pointwiseGrad = _pointwise.Backward(gradOutput);
            return _depthwise.Backward(pointwiseGrad);
        }
    }

    public class DilatedConv2D<T> : IOperation<T> where T : struct
    {
        private readonly Conv2DLayer<T> _conv;
        private readonly int _dilation;

        public DilatedConv2D(int inChannels, int outChannels, int kernelSize,
            int stride, int dilation)
        {
            _dilation = dilation;
            int padding = dilation * (kernelSize - 1) / 2;
            _conv = new Conv2DLayer<T>(inChannels, outChannels, kernelSize, stride, padding);
        }

        public Tensor<T> Forward(Tensor<T> input)
        {
            // Apply convolution with dilated kernel
            return _conv.ForwardWithDilation(input, _dilation);
        }
    }

    public class ZeroOperation<T> : IOperation<T> where T : struct
    {
        public Tensor<T> Forward(Tensor<T> input)
        {
            // Return tensor of zeros with same shape
            return new Tensor<T>(input.Shape);
        }

        public Tensor<T> Backward(Tensor<T> gradOutput)
        {
            // No gradient flows through zero operation
            return new Tensor<T>(gradOutput.Shape);
        }
    }
}
```

### Phase 2: DARTS Implementation (Week 2-3)

**Step 3: Implement Mixed Operation**

File: `src/NAS/DARTS/MixedOperation.cs`

(See Mixed Operation code in Search Spaces section)

**Step 4: Implement Search Cell**

File: `src/NAS/DARTS/SearchCell.cs`

(See SearchCell code in Search Spaces section)

**Step 5: Implement DARTS Searcher**

File: `src/NAS/DARTS/DARTSSearcher.cs`

```csharp
namespace AiDotNet.NAS.DARTS
{
    public class DARTSSearcher<T> : IArchitectureSearchStrategy<T> where T : struct
    {
        private readonly SearchSpace<T> _searchSpace;
        private readonly Tensor<T> _architectureParams;
        private readonly IOptimizer<T> _weightOptimizer;
        private readonly IOptimizer<T> _archOptimizer;

        public DARTSSearcher(SearchConfig config)
        {
            _searchSpace = new SearchSpace<T>(config);
            _architectureParams = InitializeArchitectureParams();
            _weightOptimizer = new SGD<T>(config.LearningRate);
            _archOptimizer = new Adam<T>(config.ArchitectureLearningRate);
        }

        public Architecture Search(SearchConfig config)
        {
            for (int epoch = 0; epoch < config.SearchEpochs; epoch++)
            {
                // Phase 1: Train network weights
                TrainWeights(config);

                // Phase 2: Update architecture parameters
                UpdateArchitecture(config);

                // Log progress
                if (epoch % 10 == 0)
                {
                    LogProgress(epoch);
                }
            }

            // Discretize continuous architecture
            return _searchSpace.Discretize(_architectureParams);
        }

        private void TrainWeights(SearchConfig config)
        {
            var batch = SampleBatch(_trainData, config.BatchSize);

            // Forward pass
            var output = _searchSpace.Forward(batch.X, _architectureParams);
            var loss = ComputeLoss(output, batch.Y);

            // Backward pass
            var gradients = _searchSpace.Backward(loss);

            // Update weights
            _weightOptimizer.Update(_searchSpace.Weights, gradients);
        }

        private void UpdateArchitecture(SearchConfig config)
        {
            // Compute virtual weights: w' = w - ξ∇_w L_train
            var trainGradients = ComputeTrainGradients();
            var virtualWeights = ComputeVirtualStep(_searchSpace.Weights, trainGradients);

            // Temporarily set virtual weights
            var originalWeights = _searchSpace.Weights.Clone();
            _searchSpace.SetWeights(virtualWeights);

            // Evaluate on validation data
            var valBatch = SampleBatch(_valData, config.BatchSize);
            var valOutput = _searchSpace.Forward(valBatch.X, _architectureParams);
            var valLoss = ComputeLoss(valOutput, valBatch.Y);

            // Compute ∇_α L_val(w', α)
            var archGradients = _searchSpace.BackwardArchitecture(valLoss);

            // Update architecture parameters
            _archOptimizer.Update(_architectureParams, archGradients);

            // Restore original weights
            _searchSpace.SetWeights(originalWeights);
        }
    }
}
```

### Phase 3: ENAS Implementation (Week 4-5)

**Step 6: Implement Controller**

File: `src/NAS/ENAS/LSTMController.cs`

(See LSTMController code in ENAS section)

**Step 7: Implement ENAS Searcher**

File: `src/NAS/ENAS/ENASSearcher.cs`

(See ENASSearcher code in ENAS section)

### Phase 4: Testing and Validation (Week 6)

**Step 8: Create Test Suite**

(See Testing Strategy section for comprehensive tests)

**Step 9: Run Experiments**

File: `examples/NAS/CIFAR10Search.cs`

```csharp
public class CIFAR10SearchExample
{
    public static void Run()
    {
        // Load CIFAR-10
        var (trainX, trainY, testX, testY) = LoadCIFAR10();

        // Configure search
        var config = new SearchConfig
        {
            SearchEpochs = 50,
            LearningRate = 0.025,
            ArchitectureLearningRate = 3e-4,
            BatchSize = 64
        };

        // Run DARTS search
        Console.WriteLine("Starting DARTS search...");
        var dartsSearcher = new DARTSSearcher<double>(config);
        var dartsArch = dartsSearcher.Search(config);

        Console.WriteLine("Discovered architecture:");
        dartsArch.Print();

        // Train from scratch
        var network = dartsArch.Build<double>(inputChannels: 3, numClasses: 10);
        network.Train(trainX, trainY, epochs: 600);

        // Evaluate
        var accuracy = network.Evaluate(testX, testY);
        Console.WriteLine($"Final accuracy: {accuracy:F2}%");
    }
}
```

---

## Summary

This guide provides:

1. **DARTS**: Gradient-based architecture search with continuous relaxation
2. **ENAS**: Efficient search using parameter sharing and controller
3. **Search Spaces**: Cell-based representations with primitive operations
4. **Implementation**: Complete code for both DARTS and ENAS
5. **Testing**: Comprehensive unit and integration tests

**Key Takeaways**:
- DARTS is faster but requires more memory (all operations active)
- ENAS is more memory-efficient (parameter sharing)
- Both discover competitive architectures automatically
- Cell-based search spaces generalize well across datasets

**Expected Timeline**: 6 weeks for full implementation with experiments

# Issue #405: Junior Developer Implementation Guide
## Graph Neural Networks (GCN, GAT, GraphSAGE)

---

## Table of Contents
1. [Understanding Graph Neural Networks](#understanding-graph-neural-networks)
2. [Graph Convolutional Networks (GCN)](#graph-convolutional-networks-gcn)
3. [Graph Attention Networks (GAT)](#graph-attention-networks-gat)
4. [GraphSAGE: Inductive Learning](#graphsage-inductive-learning)
5. [Graph Representations and Operations](#graph-representations-and-operations)
6. [Implementation Strategy](#implementation-strategy)
7. [Testing Strategy](#testing-strategy)
8. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)

---

## Understanding Graph Neural Networks

### What are Graph Neural Networks?

Graph Neural Networks (GNNs) are neural networks designed to operate on **graph-structured data**.

**Graph**: `G = (V, E)` where:
- `V` = Set of nodes/vertices (e.g., users, molecules, documents)
- `E` = Set of edges (e.g., friendships, chemical bonds, citations)

**Examples**:
- **Social Networks**: Nodes = users, Edges = friendships
- **Molecules**: Nodes = atoms, Edges = chemical bonds
- **Citation Networks**: Nodes = papers, Edges = citations
- **Knowledge Graphs**: Nodes = entities, Edges = relationships

### Why Graph Neural Networks?

**Traditional Neural Networks** (CNNs, RNNs) work on:
- **Grids**: Images (2D grids of pixels)
- **Sequences**: Text (1D sequences of words)

**GNNs** work on:
- **Irregular structures**: Graphs with variable numbers of neighbors
- **Non-Euclidean data**: No fixed coordinate system

### The Core Idea: Message Passing

GNNs learn node representations by **aggregating information from neighbors**:

```
For each node v:
    1. Collect messages from neighbors N(v)
    2. Aggregate messages (sum, mean, max, attention)
    3. Update node representation h_v
```

**Mathematical Form**:
```
h_v^(l+1) = σ(W * AGG({h_u^(l) : u ∈ N(v)}))
```

Where:
- `h_v^(l)` = Node v's representation at layer l
- `N(v)` = Neighbors of node v
- `AGG` = Aggregation function (sum, mean, attention)
- `W` = Learnable weight matrix
- `σ` = Activation function

### Visual Example

```
Initial node features:
   [A]--[B]
    |  X  |
   [C]--[D]

After 1 layer:
- Node A learns from neighbors: B, C
- Node B learns from neighbors: A, D
- Node C learns from neighbors: A, D
- Node D learns from neighbors: B, C

After 2 layers:
- Node A knows about all nodes (through 2-hop paths)
```

---

## Graph Convolutional Networks (GCN)

### Core Idea

Apply **spectral graph theory** to define convolutions on graphs.

Simplification: Use **symmetric normalized aggregation** for message passing.

### Mathematical Foundation

#### 1. Adjacency Matrix

Represent graph structure as matrix `A`:
```
A[i,j] = 1 if edge (i,j) exists
A[i,j] = 0 otherwise
```

Example:
```
Graph:  0---1
        |   |
        2---3

Adjacency matrix A:
    0  1  2  3
0 [ 0  1  1  0 ]
1 [ 1  0  0  1 ]
2 [ 1  0  0  1 ]
3 [ 0  1  1  0 ]
```

#### 2. Degree Matrix

Diagonal matrix `D` where `D[i,i]` = degree of node i (number of neighbors):
```
D[i,i] = Σ_j A[i,j]
```

For the example above:
```
D = diag([2, 2, 2, 2])
```

#### 3. Normalized Adjacency

Add self-loops and normalize:
```
A_hat = A + I  (add self-loops)
D_hat = degree matrix of A_hat
A_norm = D_hat^(-1/2) * A_hat * D_hat^(-1/2)
```

**Why normalize?**
- Prevents numerical instabilities
- Makes aggregation independent of node degree

#### 4. GCN Layer Formula

```
H^(l+1) = σ(A_norm * H^(l) * W^(l))
```

Where:
- `H^(l)` = Node feature matrix at layer l (N × d_l)
- `W^(l)` = Learnable weight matrix (d_l × d_{l+1})
- `A_norm` = Normalized adjacency matrix (N × N)
- `σ` = Activation function (ReLU, etc.)

**Interpretation**:
1. `H^(l) * W^(l)`: Transform features
2. `A_norm * (...)`: Aggregate from neighbors
3. `σ(...)`: Non-linearity

#### 5. Multi-Layer GCN

Stack multiple GCN layers to capture multi-hop neighborhoods:

```
H^(0) = X  (input features)
H^(1) = σ(A_norm * H^(0) * W^(0))
H^(2) = σ(A_norm * H^(1) * W^(1))
...
Z = H^(L)  (final embeddings)
```

### Implementation

```csharp
public class GCNLayer<T> : ILayer<T> where T : struct
{
    private readonly int _inputDim;
    private readonly int _outputDim;
    private Tensor<T> _weights;  // W matrix (inputDim × outputDim)
    private Tensor<T> _bias;     // bias vector (outputDim)
    private readonly IActivationFunction<T> _activation;

    // Cached values for backpropagation
    private Tensor<T> _lastInput = null!;
    private Tensor<T> _lastAdjacency = null!;

    public GCNLayer(int inputDim, int outputDim, IActivationFunction<T> activation)
    {
        _inputDim = inputDim;
        _outputDim = outputDim;
        _activation = activation;

        // Initialize weights (Xavier/Glorot initialization)
        _weights = InitializeWeights(inputDim, outputDim);
        _bias = new Tensor<T>(new[] { outputDim });
    }

    /// <summary>
    /// Forward pass: H' = σ(A_norm * H * W + b)
    /// </summary>
    public Tensor<T> Forward(Tensor<T> nodeFeatures, Tensor<T> normalizedAdjacency)
    {
        // nodeFeatures: (numNodes × inputDim)
        // normalizedAdjacency: (numNodes × numNodes)

        _lastInput = nodeFeatures;
        _lastAdjacency = normalizedAdjacency;

        // Step 1: Transform features: H * W
        var transformed = nodeFeatures.MatMul(_weights);  // (numNodes × outputDim)

        // Step 2: Aggregate from neighbors: A_norm * (H * W)
        var aggregated = normalizedAdjacency.MatMul(transformed);  // (numNodes × outputDim)

        // Step 3: Add bias
        var withBias = aggregated.Add(_bias);

        // Step 4: Apply activation
        var output = _activation.Activate(withBias);

        return output;
    }

    /// <summary>
    /// Backward pass: compute gradients
    /// </summary>
    public Tensor<T> Backward(Tensor<T> gradOutput)
    {
        // Gradient through activation
        var gradActivation = _activation.Derivative(_lastInput);
        var gradPreActivation = gradOutput.Multiply(gradActivation);

        // Gradient w.r.t. weights: H^T * A_norm^T * gradPreActivation
        var gradWeights = _lastInput.Transpose()
            .MatMul(_lastAdjacency.Transpose())
            .MatMul(gradPreActivation);

        // Gradient w.r.t. bias: sum over nodes
        var gradBias = gradPreActivation.Sum(axis: 0);

        // Gradient w.r.t. input: A_norm^T * gradPreActivation * W^T
        var gradInput = _lastAdjacency.Transpose()
            .MatMul(gradPreActivation)
            .MatMul(_weights.Transpose());

        // Update weights
        UpdateWeights(gradWeights, gradBias);

        return gradInput;
    }

    private Tensor<T> InitializeWeights(int fanIn, int fanOut)
    {
        // Xavier initialization: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
        double scale = Math.Sqrt(6.0 / (fanIn + fanOut));
        var random = new Random();

        var weights = new T[fanIn * fanOut];
        for (int i = 0; i < weights.Length; i++)
        {
            double value = (random.NextDouble() * 2 - 1) * scale;
            weights[i] = NumOps<T>.FromDouble(value);
        }

        return new Tensor<T>(weights, new[] { fanIn, fanOut });
    }
}
```

### Graph Normalization Helper

```csharp
public static class GraphNormalization
{
    /// <summary>
    /// Compute normalized adjacency: D^(-1/2) * (A + I) * D^(-1/2)
    /// </summary>
    public static Tensor<T> ComputeNormalizedAdjacency<T>(Tensor<T> adjacency) where T : struct
    {
        int numNodes = adjacency.Shape[0];

        // Step 1: Add self-loops: A_hat = A + I
        var aHat = adjacency.Clone();
        for (int i = 0; i < numNodes; i++)
        {
            aHat[i, i] = NumOps<T>.Add(aHat[i, i], NumOps<T>.One);
        }

        // Step 2: Compute degree matrix D_hat
        var degrees = new T[numNodes];
        for (int i = 0; i < numNodes; i++)
        {
            T sum = NumOps<T>.Zero;
            for (int j = 0; j < numNodes; j++)
            {
                sum = NumOps<T>.Add(sum, aHat[i, j]);
            }
            degrees[i] = sum;
        }

        // Step 3: Compute D^(-1/2)
        var degreeInvSqrt = new T[numNodes];
        for (int i = 0; i < numNodes; i++)
        {
            double d = Convert.ToDouble(degrees[i]);
            degreeInvSqrt[i] = NumOps<T>.FromDouble(1.0 / Math.Sqrt(d));
        }

        // Step 4: Compute D^(-1/2) * A_hat * D^(-1/2)
        var normalized = new Tensor<T>(new[] { numNodes, numNodes });
        for (int i = 0; i < numNodes; i++)
        {
            for (int j = 0; j < numNodes; j++)
            {
                var value = NumOps<T>.Multiply(
                    NumOps<T>.Multiply(degreeInvSqrt[i], aHat[i, j]),
                    degreeInvSqrt[j]
                );
                normalized[i, j] = value;
            }
        }

        return normalized;
    }
}
```

---

## Graph Attention Networks (GAT)

### Core Idea

Use **attention mechanism** to learn importance weights for neighbors.

**Problem with GCN**: All neighbors have equal importance (after normalization).

**GAT Solution**: Learn attention weights `α_{ij}` for each edge (i → j).

### Mathematical Foundation

#### 1. Attention Mechanism

For each edge (i → j), compute attention coefficient:

```
e_{ij} = LeakyReLU(a^T [W*h_i || W*h_j])
```

Where:
- `h_i`, `h_j` = Node features for nodes i and j
- `W` = Shared weight matrix
- `a` = Attention vector (learnable)
- `||` = Concatenation
- `LeakyReLU` = Activation function

#### 2. Normalize with Softmax

```
α_{ij} = softmax_j(e_{ij}) = exp(e_{ij}) / Σ_{k∈N(i)} exp(e_{ik})
```

**Result**: `α_{ij}` represents the importance of node j to node i.

#### 3. Aggregate with Attention

```
h_i' = σ(Σ_{j∈N(i)} α_{ij} * W * h_j)
```

#### 4. Multi-Head Attention

Use multiple attention heads (like Transformer):

```
h_i' = ||_{k=1}^K σ(Σ_{j∈N(i)} α_{ij}^k * W^k * h_j)
```

Where:
- `K` = Number of attention heads
- `||` = Concatenation or averaging

### Implementation

```csharp
public class GATLayer<T> : ILayer<T> where T : struct
{
    private readonly int _inputDim;
    private readonly int _outputDim;
    private readonly int _numHeads;
    private readonly List<AttentionHead<T>> _heads;
    private readonly bool _concat;  // Concatenate or average heads

    public GATLayer(int inputDim, int outputDim, int numHeads = 8, bool concat = true)
    {
        _inputDim = inputDim;
        _outputDim = outputDim;
        _numHeads = numHeads;
        _concat = concat;

        // Create multiple attention heads
        _heads = new List<AttentionHead<T>>();
        for (int i = 0; i < numHeads; i++)
        {
            _heads.Add(new AttentionHead<T>(inputDim, outputDim / numHeads));
        }
    }

    public Tensor<T> Forward(Tensor<T> nodeFeatures, Tensor<T> adjacency)
    {
        // Compute attention for each head
        var headOutputs = _heads.Select(head =>
            head.Forward(nodeFeatures, adjacency)
        ).ToList();

        // Combine heads
        if (_concat)
        {
            // Concatenate along feature dimension
            return Tensor<T>.Concatenate(headOutputs, axis: 1);
        }
        else
        {
            // Average across heads
            var sum = headOutputs[0];
            for (int i = 1; i < headOutputs.Count; i++)
            {
                sum = sum.Add(headOutputs[i]);
            }
            return sum.Divide(NumOps<T>.FromDouble(_numHeads));
        }
    }
}

public class AttentionHead<T> where T : struct
{
    private Tensor<T> _W;  // Weight matrix
    private Tensor<T> _a;  // Attention vector
    private readonly double _leakyReluSlope = 0.2;

    private Tensor<T> _lastFeatures = null!;
    private Tensor<T> _lastAdjacency = null!;
    private Tensor<T> _lastAlphas = null!;

    public AttentionHead(int inputDim, int outputDim)
    {
        _W = InitializeWeights(inputDim, outputDim);
        _a = InitializeWeights(2 * outputDim, 1);  // [W*h_i || W*h_j] has dim 2*outputDim
    }

    public Tensor<T> Forward(Tensor<T> nodeFeatures, Tensor<T> adjacency)
    {
        // nodeFeatures: (numNodes × inputDim)
        // adjacency: (numNodes × numNodes)

        _lastFeatures = nodeFeatures;
        _lastAdjacency = adjacency;

        int numNodes = nodeFeatures.Shape[0];

        // Step 1: Transform features: h' = W * h
        var transformed = nodeFeatures.MatMul(_W);  // (numNodes × outputDim)

        // Step 2: Compute attention coefficients
        var attentionScores = ComputeAttentionScores(transformed, adjacency);

        // Step 3: Apply softmax to get attention weights
        var alphas = ComputeAttentionWeights(attentionScores, adjacency);
        _lastAlphas = alphas;

        // Step 4: Aggregate features using attention
        var aggregated = alphas.MatMul(transformed);  // (numNodes × outputDim)

        // Step 5: Apply activation (ELU is common for GAT)
        return ApplyELU(aggregated);
    }

    private Tensor<T> ComputeAttentionScores(Tensor<T> features, Tensor<T> adjacency)
    {
        int numNodes = features.Shape[0];
        var scores = new Tensor<T>(new[] { numNodes, numNodes });

        for (int i = 0; i < numNodes; i++)
        {
            for (int j = 0; j < numNodes; j++)
            {
                // Only compute attention for existing edges
                if (Convert.ToDouble(adjacency[i, j]) == 0 && i != j)
                {
                    scores[i, j] = NumOps<T>.FromDouble(double.NegativeInfinity);
                    continue;
                }

                // Concatenate [h_i || h_j]
                var concat = ConcatenateFeatures(features, i, j);

                // e_ij = LeakyReLU(a^T * [W*h_i || W*h_j])
                var score = concat.MatMul(_a).ToScalar();
                scores[i, j] = ApplyLeakyReLU(score, _leakyReluSlope);
            }
        }

        return scores;
    }

    private Tensor<T> ComputeAttentionWeights(Tensor<T> scores, Tensor<T> adjacency)
    {
        int numNodes = scores.Shape[0];
        var alphas = new Tensor<T>(new[] { numNodes, numNodes });

        // Softmax for each node's neighbors
        for (int i = 0; i < numNodes; i++)
        {
            // Collect scores for node i's neighbors
            var neighborScores = new List<double>();
            var neighborIndices = new List<int>();

            for (int j = 0; j < numNodes; j++)
            {
                if (Convert.ToDouble(adjacency[i, j]) > 0 || i == j)
                {
                    neighborScores.Add(Convert.ToDouble(scores[i, j]));
                    neighborIndices.Add(j);
                }
            }

            // Apply softmax
            var softmaxValues = Softmax(neighborScores);

            // Assign attention weights
            for (int k = 0; k < neighborIndices.Count; k++)
            {
                int j = neighborIndices[k];
                alphas[i, j] = NumOps<T>.FromDouble(softmaxValues[k]);
            }
        }

        return alphas;
    }

    private double[] Softmax(List<double> scores)
    {
        double max = scores.Max();
        var exp = scores.Select(s => Math.Exp(s - max)).ToArray();
        double sum = exp.Sum();
        return exp.Select(e => e / sum).ToArray();
    }

    private T ApplyLeakyReLU(T x, double slope)
    {
        double value = Convert.ToDouble(x);
        return NumOps<T>.FromDouble(value > 0 ? value : slope * value);
    }
}
```

---

## GraphSAGE: Inductive Learning

### Core Idea

**Problem with GCN/GAT**: They are **transductive** - cannot generalize to unseen nodes.
- Require the full graph during training
- Cannot handle new nodes at test time

**GraphSAGE Solution**: Learn aggregation functions that generalize to new nodes.

### Key Innovation: Sampling and Aggregating

Instead of using all neighbors:
1. **Sample** a fixed number of neighbors
2. **Aggregate** their features using a learned function
3. **Combine** with node's own features

### Mathematical Foundation

#### 1. Neighbor Sampling

```
N_k(v) = Sample k neighbors from N(v)
```

**Why sample?**
- Computational efficiency (avoid exponential blowup)
- Fixed mini-batch sizes
- Regularization effect

#### 2. Aggregation Functions

**Mean Aggregator**:
```
h_{N(v)}^k = MEAN({h_u^{k-1} : u ∈ N_k(v)})
```

**LSTM Aggregator**:
```
h_{N(v)}^k = LSTM({h_u^{k-1} : u ∈ N_k(v)})
```

**Pooling Aggregator**:
```
h_{N(v)}^k = MAX({σ(W * h_u^{k-1} + b) : u ∈ N_k(v)})
```

#### 3. Combining with Self

```
h_v^k = σ(W^k * CONCAT(h_v^{k-1}, h_{N(v)}^k))
```

Or with skip connections:
```
h_v^k = σ(W^k * h_{N(v)}^k + h_v^{k-1})
```

#### 4. Normalization

L2-normalize embeddings:
```
h_v^k ← h_v^k / ||h_v^k||_2
```

### GraphSAGE Algorithm

```
Algorithm: GraphSAGE Forward
Input: Graph G(V, E), node features {x_v : v ∈ V}, depth K
Output: Node embeddings {z_v : v ∈ V}

1. Initialize: h_v^0 ← x_v for all v

2. For k = 1 to K:
   For each node v in V:
      a. Sample neighbors: N_k(v) ← Sample(N(v), k)

      b. Aggregate neighbor features:
         h_{N(v)}^k ← AGGREGATE({h_u^{k-1} : u ∈ N_k(v)})

      c. Combine with self:
         h_v^k ← σ(W^k * CONCAT(h_v^{k-1}, h_{N(v)}^k))

      d. Normalize:
         h_v^k ← h_v^k / ||h_v^k||_2

3. Return: z_v = h_v^K for all v
```

### Implementation

```csharp
public class GraphSAGELayer<T> : ILayer<T> where T : struct
{
    private readonly int _inputDim;
    private readonly int _outputDim;
    private readonly int _numSamples;  // Number of neighbors to sample
    private readonly AggregatorType _aggregatorType;
    private Tensor<T> _weights;
    private readonly IActivationFunction<T> _activation;

    public enum AggregatorType
    {
        Mean,
        LSTM,
        Pool,
        GCN
    }

    public GraphSAGELayer(
        int inputDim,
        int outputDim,
        int numSamples = 25,
        AggregatorType aggregatorType = AggregatorType.Mean)
    {
        _inputDim = inputDim;
        _outputDim = outputDim;
        _numSamples = numSamples;
        _aggregatorType = aggregatorType;
        _activation = new ReLU<T>();

        // Weight matrix for combining self + neighbor features
        _weights = InitializeWeights(2 * inputDim, outputDim);
    }

    public Tensor<T> Forward(Tensor<T> nodeFeatures, Tensor<T> adjacency)
    {
        int numNodes = nodeFeatures.Shape[0];
        var output = new Tensor<T>(new[] { numNodes, _outputDim });

        for (int nodeIdx = 0; nodeIdx < numNodes; nodeIdx++)
        {
            // Step 1: Sample neighbors
            var neighborIndices = SampleNeighbors(adjacency, nodeIdx, _numSamples);

            // Step 2: Aggregate neighbor features
            var aggregated = AggregateNeighbors(nodeFeatures, neighborIndices);

            // Step 3: Concatenate with self features
            var selfFeatures = nodeFeatures.GetRow(nodeIdx);
            var combined = Concatenate(selfFeatures, aggregated);

            // Step 4: Transform and activate
            var transformed = combined.MatMul(_weights);
            var activated = _activation.Activate(transformed);

            // Step 5: L2 normalize
            var normalized = L2Normalize(activated);

            output.SetRow(nodeIdx, normalized);
        }

        return output;
    }

    private List<int> SampleNeighbors(Tensor<T> adjacency, int nodeIdx, int k)
    {
        // Find all neighbors
        var neighbors = new List<int>();
        int numNodes = adjacency.Shape[0];

        for (int j = 0; j < numNodes; j++)
        {
            if (Convert.ToDouble(adjacency[nodeIdx, j]) > 0 && j != nodeIdx)
            {
                neighbors.Add(j);
            }
        }

        // Sample k neighbors (with replacement if needed)
        if (neighbors.Count == 0)
            return new List<int> { nodeIdx };  // Use self if no neighbors

        var sampled = new List<int>();
        var random = new Random();

        for (int i = 0; i < Math.Min(k, neighbors.Count); i++)
        {
            int idx = random.Next(neighbors.Count);
            sampled.Add(neighbors[idx]);
        }

        return sampled;
    }

    private Tensor<T> AggregateNeighbors(Tensor<T> nodeFeatures, List<int> neighborIndices)
    {
        return _aggregatorType switch
        {
            AggregatorType.Mean => AggregateMean(nodeFeatures, neighborIndices),
            AggregatorType.Pool => AggregatePool(nodeFeatures, neighborIndices),
            AggregatorType.LSTM => AggregateLSTM(nodeFeatures, neighborIndices),
            _ => AggregateMean(nodeFeatures, neighborIndices)
        };
    }

    private Tensor<T> AggregateMean(Tensor<T> nodeFeatures, List<int> indices)
    {
        if (indices.Count == 0)
            return new Tensor<T>(new[] { _inputDim });

        // Compute mean of neighbor features
        var sum = new Tensor<T>(new[] { _inputDim });

        foreach (var idx in indices)
        {
            var features = nodeFeatures.GetRow(idx);
            sum = sum.Add(features);
        }

        return sum.Divide(NumOps<T>.FromDouble(indices.Count));
    }

    private Tensor<T> AggregatePool(Tensor<T> nodeFeatures, List<int> indices)
    {
        if (indices.Count == 0)
            return new Tensor<T>(new[] { _inputDim });

        // Transform each neighbor then take element-wise max
        var pooled = new T[_inputDim];

        for (int dim = 0; dim < _inputDim; dim++)
        {
            T maxValue = NumOps<T>.FromDouble(double.NegativeInfinity);

            foreach (var idx in indices)
            {
                var value = nodeFeatures[idx, dim];
                if (NumOps<T>.GreaterThan(value, maxValue))
                {
                    maxValue = value;
                }
            }

            pooled[dim] = maxValue;
        }

        return new Tensor<T>(pooled, new[] { _inputDim });
    }

    private Tensor<T> L2Normalize(Tensor<T> vector)
    {
        // Compute L2 norm
        double sumSquares = 0;
        var data = vector.ToArray();

        foreach (var value in data)
        {
            double v = Convert.ToDouble(value);
            sumSquares += v * v;
        }

        double norm = Math.Sqrt(sumSquares);
        if (norm < 1e-12)
            return vector;  // Avoid division by zero

        // Normalize
        var normalized = new T[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            normalized[i] = NumOps<T>.FromDouble(Convert.ToDouble(data[i]) / norm);
        }

        return new Tensor<T>(normalized, vector.Shape);
    }
}
```

---

## Graph Representations and Operations

### Graph Data Structure

```csharp
public class Graph<T> where T : struct
{
    public Tensor<T> NodeFeatures { get; set; }  // (numNodes × featureDim)
    public Tensor<T> EdgeIndex { get; set; }     // (2 × numEdges) - edge list format
    public Tensor<T> EdgeWeights { get; set; }   // (numEdges) - optional edge weights
    public int NumNodes { get; set; }
    public int NumEdges { get; set; }

    /// <summary>
    /// Convert edge list to adjacency matrix.
    /// </summary>
    public Tensor<T> ToAdjacencyMatrix()
    {
        var adjacency = new Tensor<T>(new[] { NumNodes, NumNodes });

        for (int e = 0; e < NumEdges; e++)
        {
            int src = Convert.ToInt32(EdgeIndex[0, e]);
            int dst = Convert.ToInt32(EdgeIndex[1, e]);
            T weight = EdgeWeights != null ? EdgeWeights[e] : NumOps<T>.One;

            adjacency[src, dst] = weight;
            // For undirected graphs:
            // adjacency[dst, src] = weight;
        }

        return adjacency;
    }

    /// <summary>
    /// Get neighbors of a node.
    /// </summary>
    public List<int> GetNeighbors(int nodeIdx)
    {
        var neighbors = new List<int>();

        for (int e = 0; e < NumEdges; e++)
        {
            int src = Convert.ToInt32(EdgeIndex[0, e]);
            int dst = Convert.ToInt32(EdgeIndex[1, e]);

            if (src == nodeIdx)
                neighbors.Add(dst);
        }

        return neighbors;
    }
}
```

### Graph Batch Processing

```csharp
public class GraphBatch<T> where T : struct
{
    public List<Graph<T>> Graphs { get; set; } = new List<Graph<T>>();

    /// <summary>
    /// Combine multiple graphs into a single disconnected graph.
    /// </summary>
    public Graph<T> Collate()
    {
        int totalNodes = 0;
        int totalEdges = 0;

        // Count totals
        foreach (var graph in Graphs)
        {
            totalNodes += graph.NumNodes;
            totalEdges += graph.NumEdges;
        }

        // Allocate combined tensors
        int featureDim = Graphs[0].NodeFeatures.Shape[1];
        var nodeFeatures = new List<T[]>();
        var edgeList = new List<(int, int)>();

        int nodeOffset = 0;

        // Concatenate graphs
        foreach (var graph in Graphs)
        {
            // Add node features
            for (int i = 0; i < graph.NumNodes; i++)
            {
                nodeFeatures.Add(graph.NodeFeatures.GetRow(i).ToArray());
            }

            // Add edges (with offset)
            for (int e = 0; e < graph.NumEdges; e++)
            {
                int src = Convert.ToInt32(graph.EdgeIndex[0, e]) + nodeOffset;
                int dst = Convert.ToInt32(graph.EdgeIndex[1, e]) + nodeOffset;
                edgeList.Add((src, dst));
            }

            nodeOffset += graph.NumNodes;
        }

        // Create combined graph
        var combinedFeatures = new Tensor<T>(
            nodeFeatures.SelectMany(f => f).ToArray(),
            new[] { totalNodes, featureDim }
        );

        var combinedEdgeIndex = new Tensor<T>(new[] { 2, totalEdges });
        for (int e = 0; e < totalEdges; e++)
        {
            combinedEdgeIndex[0, e] = NumOps<T>.FromDouble(edgeList[e].Item1);
            combinedEdgeIndex[1, e] = NumOps<T>.FromDouble(edgeList[e].Item2);
        }

        return new Graph<T>
        {
            NodeFeatures = combinedFeatures,
            EdgeIndex = combinedEdgeIndex,
            NumNodes = totalNodes,
            NumEdges = totalEdges
        };
    }
}
```

---

## Testing Strategy

### Unit Tests

```csharp
[TestClass]
public class GNNTests
{
    [TestMethod]
    public void GCNLayer_ForwardPass_CorrectShape()
    {
        // Arrange: Create simple graph (4 nodes, fully connected)
        var adjacency = CreateFullyConnectedGraph(numNodes: 4);
        var normalizedAdj = GraphNormalization.ComputeNormalizedAdjacency(adjacency);

        var nodeFeatures = CreateRandomTensor(4, 8);  // 4 nodes, 8 features
        var gcnLayer = new GCNLayer<double>(inputDim: 8, outputDim: 16, new ReLU<double>());

        // Act
        var output = gcnLayer.Forward(nodeFeatures, normalizedAdj);

        // Assert: Output shape is (4 × 16)
        Assert.AreEqual(4, output.Shape[0]);
        Assert.AreEqual(16, output.Shape[1]);
    }

    [TestMethod]
    public void GAT_AttentionWeights_SumToOne()
    {
        // Arrange
        var graph = CreateSimpleGraph();  // 3 nodes: 0-1, 1-2
        var gatLayer = new GATLayer<double>(inputDim: 4, outputDim: 8, numHeads: 2);

        // Act
        var output = gatLayer.Forward(graph.NodeFeatures, graph.ToAdjacencyMatrix());

        // Get attention weights for node 1 (has 2 neighbors)
        var alphas = gatLayer.GetLastAttentionWeights();

        // Assert: Attention weights for each node sum to 1
        for (int node = 0; node < 3; node++)
        {
            double sum = 0;
            for (int neighbor = 0; neighbor < 3; neighbor++)
            {
                sum += Convert.ToDouble(alphas[node, neighbor]);
            }
            Assert.AreEqual(1.0, sum, 1e-6);
        }
    }

    [TestMethod]
    public void GraphSAGE_Sampling_ReturnsCorrectCount()
    {
        // Arrange: Node with 10 neighbors
        var graph = CreateStarGraph(numNeighbors: 10);  // Center node + 10 neighbors
        var sageLayer = new GraphSAGELayer<double>(
            inputDim: 4, outputDim: 8, numSamples: 5
        );

        // Act
        var output = sageLayer.Forward(graph.NodeFeatures, graph.ToAdjacencyMatrix());

        // Assert: Should sample exactly 5 neighbors
        var sampledIndices = sageLayer.GetLastSampledNeighbors(centerNode: 0);
        Assert.AreEqual(5, sampledIndices.Count);
    }

    [TestMethod]
    public void GraphSAGE_Inductive_WorksOnNewNodes()
    {
        // Arrange: Train on graph A
        var trainGraph = CreateGraph(numNodes: 100);
        var sageModel = new GraphSAGE<double>(layerDims: new[] { 16, 32, 64 });
        sageModel.Train(trainGraph);

        // Act: Test on graph B (completely new nodes)
        var testGraph = CreateGraph(numNodes: 50);  // Different nodes!
        var embeddings = sageModel.Forward(testGraph);

        // Assert: Should produce embeddings for new nodes
        Assert.AreEqual(50, embeddings.Shape[0]);
        Assert.AreEqual(64, embeddings.Shape[1]);  // Output dimension
    }
}
```

### Integration Tests

```csharp
[TestClass]
public class GNNIntegrationTests
{
    [TestMethod]
    public void GCN_Cora_NodeClassification()
    {
        // Arrange: Load Cora citation network
        var (graph, labels, trainMask, testMask) = LoadCoraDataset();

        // Build 2-layer GCN
        var gcn = new GCN<double>(
            inputDim: graph.NodeFeatures.Shape[1],
            hiddenDim: 16,
            outputDim: 7,  // 7 classes in Cora
            numLayers: 2
        );

        // Act: Train
        gcn.Train(graph, labels, trainMask, epochs: 200);

        // Evaluate
        var predictions = gcn.Forward(graph);
        var accuracy = ComputeAccuracy(predictions, labels, testMask);

        // Assert: Should achieve > 70% accuracy
        Assert.IsTrue(accuracy > 0.70, $"Expected > 70%, got {accuracy:F2}%");
    }

    [TestMethod]
    public void GAT_TransductiveLearning_ConvergesCorrectly()
    {
        // Arrange
        var (graph, labels) = LoadPubMedDataset();
        var gat = new GAT<double>(
            inputDim: graph.NodeFeatures.Shape[1],
            hiddenDim: 8,
            outputDim: 3,
            numHeads: 8
        );

        // Act
        var losses = gat.TrainWithLogging(graph, labels, epochs: 100);

        // Assert: Loss should decrease
        Assert.IsTrue(losses.Last() < losses.First());
        Assert.IsTrue(losses.Last() < 0.5);
    }

    [TestMethod]
    public void GraphSAGE_ProteinProteinInteraction_InductiveLearning()
    {
        // Arrange: Load PPI dataset (inductive setting)
        var (trainGraphs, trainLabels) = LoadPPITrain();
        var (testGraphs, testLabels) = LoadPPITest();  // Completely new graphs!

        var sage = new GraphSAGE<double>(
            inputDim: 50,
            hiddenDims: new[] { 256, 256, 128 },
            numSamples: new[] { 25, 10 }
        );

        // Act: Train on training graphs
        sage.Train(trainGraphs, trainLabels, epochs: 100);

        // Test on completely new graphs (inductive)
        var predictions = sage.PredictBatch(testGraphs);
        var f1Score = ComputeF1Score(predictions, testLabels);

        // Assert: Should achieve > 0.5 F1 score
        Assert.IsTrue(f1Score > 0.5);
    }
}
```

---

## Step-by-Step Implementation Guide

### Phase 1: Graph Infrastructure (Week 1)

**Step 1: Create Graph Data Structure**

File: `src/GraphNeuralNetworks/Graph.cs`

(See Graph Representations section for full implementation)

**Step 2: Implement Graph Utilities**

File: `src/GraphNeuralNetworks/GraphUtils.cs`

```csharp
public static class GraphUtils
{
    public static Tensor<T> ComputeNormalizedAdjacency<T>(Tensor<T> adjacency)
    {
        // See GraphNormalization.ComputeNormalizedAdjacency implementation
    }

    public static Tensor<T> EdgeListToAdjacencyMatrix<T>(Tensor<T> edgeIndex, int numNodes)
    {
        // Convert edge list format to adjacency matrix
    }

    public static List<int> GetKHopNeighbors<T>(Graph<T> graph, int nodeIdx, int k)
    {
        // Get all neighbors within k hops
    }
}
```

### Phase 2: GCN Implementation (Week 2)

**Step 3: Implement GCN Layer**

File: `src/GraphNeuralNetworks/Layers/GCNLayer.cs`

(See GCN section for full implementation)

**Step 4: Build Multi-Layer GCN**

File: `src/GraphNeuralNetworks/Models/GCN.cs`

```csharp
public class GCN<T> : NeuralNetwork<T> where T : struct
{
    private readonly List<GCNLayer<T>> _layers;
    private readonly Tensor<T> _normalizedAdjacency;

    public GCN(int inputDim, int hiddenDim, int outputDim, int numLayers = 2)
    {
        _layers = new List<GCNLayer<T>>();

        // Input layer
        _layers.Add(new GCNLayer<T>(inputDim, hiddenDim, new ReLU<T>()));

        // Hidden layers
        for (int i = 1; i < numLayers - 1; i++)
        {
            _layers.Add(new GCNLayer<T>(hiddenDim, hiddenDim, new ReLU<T>()));
        }

        // Output layer
        _layers.Add(new GCNLayer<T>(hiddenDim, outputDim, new Softmax<T>()));
    }

    public Tensor<T> Forward(Graph<T> graph)
    {
        var adjacency = graph.ToAdjacencyMatrix();
        var normalizedAdj = GraphUtils.ComputeNormalizedAdjacency(adjacency);

        var h = graph.NodeFeatures;

        foreach (var layer in _layers)
        {
            h = layer.Forward(h, normalizedAdj);
        }

        return h;
    }
}
```

### Phase 3: GAT Implementation (Week 3)

**Step 5: Implement Attention Head**

File: `src/GraphNeuralNetworks/Layers/AttentionHead.cs`

(See GAT section for full implementation)

**Step 6: Implement GAT Layer**

File: `src/GraphNeuralNetworks/Layers/GATLayer.cs`

(See GAT section for full implementation)

### Phase 4: GraphSAGE Implementation (Week 4)

**Step 7: Implement Aggregators**

File: `src/GraphNeuralNetworks/Aggregators/IAggregator.cs`

```csharp
public interface IAggregator<T> where T : struct
{
    Tensor<T> Aggregate(Tensor<T> nodeFeatures, List<int> neighborIndices);
}

public class MeanAggregator<T> : IAggregator<T> where T : struct
{
    // See AggregateMean in GraphSAGE section
}

public class PoolAggregator<T> : IAggregator<T> where T : struct
{
    // See AggregatePool in GraphSAGE section
}
```

**Step 8: Implement GraphSAGE Layer**

File: `src/GraphNeuralNetworks/Layers/GraphSAGELayer.cs`

(See GraphSAGE section for full implementation)

### Phase 5: Testing and Validation (Week 5-6)

**Step 9: Create Test Datasets**

File: `tests/GraphNeuralNetworks/TestData/GraphDatasets.cs`

```csharp
public static class GraphDatasets
{
    public static (Graph<double>, Tensor<double>, bool[], bool[]) LoadCora()
    {
        // Load Cora citation network
        // Returns: (graph, labels, trainMask, testMask)
    }

    public static Graph<double> CreateKarateClub()
    {
        // Create Zachary's Karate Club graph (classic test case)
    }
}
```

**Step 10: Run Comprehensive Tests**

(See Testing Strategy section for complete test suite)

---

## Summary

This guide provides:

1. **GCN**: Spectral graph convolutions with normalized aggregation
2. **GAT**: Attention-based neighbor aggregation for adaptive weighting
3. **GraphSAGE**: Inductive learning via sampling and aggregation
4. **Graph Operations**: Efficient representations and batch processing
5. **Testing**: Comprehensive unit and integration tests on real datasets

**Key Differences**:
- **GCN**: Fast, simple, works well on homophilic graphs (similar neighbors)
- **GAT**: Learns attention weights, better for heterophilic graphs
- **GraphSAGE**: Inductive (generalizes to new nodes), sampling-based

**Expected Timeline**: 6 weeks for full implementation with extensive testing

# Issue #311: Mixture-of-Experts (MoE) Architecture - Implementation Guide

## For Junior Developers: Complete Implementation Tutorial

### Table of Contents
1. [Understanding Mixture-of-Experts](#understanding-moe)
2. [Why MoE Scales Better Than Dense Models](#advantages)
3. [Architecture Deep Dive](#architecture)
4. [The Load Balancing Challenge](#load-balancing)
5. [Implementation Guide](#implementation-guide)
6. [Testing Strategy](#testing-strategy)
7. [Common Pitfalls](#common-pitfalls)

---

## Understanding MoE

### What is Mixture-of-Experts?

**For Beginners:** Mixture-of-Experts (MoE) is like having a team of specialists instead of one generalist. Each "expert" is good at handling specific types of inputs, and a "router" (or gating network) decides which experts should process each input.

**Real-world analogy:**
- **Dense Model**: Like having one doctor handle all medical cases
- **MoE Model**: Like a hospital with specialists (cardiologist, neurologist, etc.) and a triage system (router) that sends patients to the right specialist

**Key insight:** You can have MANY experts (increasing model capacity) but only activate a FEW experts per input (keeping computation manageable).

**Example:**
```
Input: "The capital of France is..."
Router: This is a geography question → Send to Expert 3 (geography specialist)

Input: "To solve x² + 5x + 6 = 0, we..."
Router: This is a math question → Send to Expert 7 (math specialist)
```

### How It Works

**Traditional dense layer:**
```csharp
// All neurons process all inputs
Vector<T> output = DenseLayer.Forward(input);
// Computation: O(input_dim × output_dim)
```

**MoE layer:**
```csharp
// 1. Router selects top-k experts for each input
int[] expertIndices = Router.SelectTopK(input, k: 2);

// 2. Only those experts process the input
Vector<T> output = CombineExpertOutputs(
    Expert[expertIndices[0]].Forward(input),
    Expert[expertIndices[1]].Forward(input));

// Computation: O(k × expert_size) where k << num_experts
```

---

## Advantages

### Why Use Mixture-of-Experts?

#### 1. Massive Capacity with Controlled Compute

**Problem with scaling dense models:**
```csharp
// Dense model with 10 billion parameters
// Every forward pass uses ALL 10 billion parameters
DenseLayer huge = new DenseLayer(10_000, 1_000_000); // 10B params
Vector<T> output = huge.Forward(input); // ALL params active!
```

**MoE solution:**
```csharp
// MoE with 8 experts, each 1.25B params = 10B total
// But only use 2 experts per input = 2.5B params active
MixtureOfExpertsLayer moe = new MixtureOfExpertsLayer(
    numExperts: 8,
    expertSize: 1_250_000,
    topK: 2);

// Only 25% of parameters active per input!
Vector<T> output = moe.Forward(input);
```

**Benefits:**
- **Training**: Can train much larger models with same compute
- **Inference**: Faster than equivalent dense model
- **Memory**: Only need to keep active experts in fast memory

#### 2. Automatic Specialization

**The model learns to specialize automatically!**

```csharp
// During training, experts naturally specialize:
// Expert 0: Code and programming
// Expert 1: Science and math
// Expert 2: History and geography
// Expert 3: Arts and literature
// ... etc

// Router learns to route inputs to the right expert
```

**Real example from GPT-4 (which uses MoE):**
- Model has ~1.8 trillion parameters (220+ experts)
- Each forward pass only uses ~280 billion (top-16 experts)
- But has capacity of the full 1.8T parameter model!

#### 3. Efficiency Comparison

**Computation for processing one token:**

**Dense Model (1T parameters):**
```
Operations: 2 × 1T = 2 trillion FLOPs
Memory: 1T × 2 bytes (fp16) = 2TB
```

**MoE Model (1T parameters, 8 experts, top-2):**
```
Router overhead: 2 × embedding_dim = ~2 million FLOPs
Expert computation: 2 × (1T/8) = 250 billion FLOPs
Total: ~250 billion FLOPs (8× faster!)
Memory (active): 250B × 2 bytes = 500GB (4× less!)
```

---

## Architecture

### Components of an MoE Layer

```
Input Tokens
    ↓
┌─────────────────────────┐
│   Router / Gating       │  ← Learns which expert(s) to use
│   Network               │
└─────────────────────────┘
    ↓
[probabilities for each expert]
    ↓
Select Top-K Experts
    ↓
┌──────┬──────┬──────┬──────┐
│Exp 0 │Exp 1 │ ...  │Exp N │  ← Only top-k are activated
└──────┴──────┴──────┴──────┘
    ↓
Combine Expert Outputs (weighted by router probabilities)
    ↓
Output
```

### The Router (Gating Network)

**Purpose:** Decide which expert(s) should process each input.

**Simple implementation:**
```csharp
public class Router<T>
{
    private readonly DenseLayer<T> _gatingLayer;
    private readonly int _numExperts;
    private readonly int _topK;

    /// <summary>
    /// Computes router probabilities and selects top-k experts.
    /// </summary>
    /// <param name="input">Input token: (batch, seq_len, d_model)</param>
    /// <returns>Expert indices and probabilities for each token</returns>
    public (int[][] expertIndices, T[][] expertWeights) Route(Tensor<T> input)
    {
        // input: (batch, seq_len, d_model)
        int batch = input.Shape[0];
        int seqLen = input.Shape[1];

        // Project to expert logits: (batch, seq_len, num_experts)
        Tensor<T> logits = _gatingLayer.Forward(input);

        // Apply softmax to get probabilities
        Tensor<T> probs = Softmax(logits, axis: 2);

        // Select top-k experts for each token
        int[][] expertIndices = new int[batch * seqLen][];
        T[][] expertWeights = new T[batch * seqLen][];

        int tokenIdx = 0;
        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                // Get probabilities for this token
                T[] tokenProbs = new T[_numExperts];
                for (int e = 0; e < _numExperts; e++)
                    tokenProbs[e] = probs[b, t, e];

                // Find top-k experts
                var topK = GetTopK(tokenProbs, _topK);
                expertIndices[tokenIdx] = topK.indices;
                expertWeights[tokenIdx] = topK.weights;

                // Renormalize weights to sum to 1
                T sum = NumOps.Zero;
                foreach (var w in expertWeights[tokenIdx])
                    sum = NumOps.Add(sum, w);

                for (int k = 0; k < _topK; k++)
                    expertWeights[tokenIdx][k] = NumOps.Divide(
                        expertWeights[tokenIdx][k], sum);

                tokenIdx++;
            }
        }

        return (expertIndices, expertWeights);
    }
}
```

**For Beginners:**
1. The router is just a dense layer followed by softmax
2. Softmax converts logits to probabilities (sum to 1)
3. Top-k selection picks the k experts with highest probability
4. Weights are renormalized so top-k weights sum to 1

### The Expert

**An expert is a standard neural network module** - typically a feed-forward network.

```csharp
public class Expert<T> : LayerBase<T>
{
    private readonly DenseLayer<T> _layer1;
    private readonly DenseLayer<T> _layer2;
    private readonly IActivationFunction<T> _activation;

    /// <summary>
    /// Creates an expert network (standard FFN).
    /// </summary>
    /// <param name="inputSize">Input dimension (e.g., 512)</param>
    /// <param name="hiddenSize">Hidden dimension (e.g., 2048).
    /// <b>Default:</b> 4× input size (standard in Transformers).</param>
    public Expert(int inputSize, int? hiddenSize = null)
        : base(new[] { inputSize }, new[] { inputSize })
    {
        int hidden = hiddenSize ?? (inputSize * 4);

        _layer1 = new DenseLayer<T>(inputSize, hidden);
        _layer2 = new DenseLayer<T>(hidden, inputSize);
        _activation = new ReLUActivation<T>();
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Standard feed-forward network
        var h = _layer1.Forward(input);
        h = h.Transform((val, _) => _activation.Activate(val));
        var output = _layer2.Forward(h);
        return output;
    }

    // ... other methods
}
```

**For Beginners:** An expert is just a mini-network. In MoE, you have many of these mini-networks in parallel.

### Token Dispatching and Combining

**The Complex Part:** Efficiently routing tokens to experts and combining results.

**Challenge:**
```
Batch of 3 sequences, each 4 tokens = 12 total tokens
8 experts, top-2 routing

Token 0: → Experts [2, 5]
Token 1: → Experts [2, 7]
Token 2: → Experts [1, 5]
...

Expert 2 needs to process: Tokens [0, 1, 7, 10] (random positions!)
Expert 5 needs to process: Tokens [0, 2, 4, 8, 11]
```

**Solution: Batching by expert**
```csharp
// 1. Group tokens by expert
Dictionary<int, List<(int tokenIdx, T weight)>> expertToTokens;

// 2. Process each expert's batch
foreach (var (expertIdx, tokens) in expertToTokens)
{
    // Gather inputs for this expert
    Tensor<T> expertInput = GatherTokens(tokens);

    // Process batch through expert
    Tensor<T> expertOutput = experts[expertIdx].Forward(expertInput);

    // Scatter outputs back to original positions (weighted)
    ScatterOutputs(expertOutput, tokens);
}

// 3. Combine results (sum weighted outputs for each token)
```

---

## Load Balancing

### The Routing Collapse Problem

**Problem:** Without constraints, the router might send all tokens to just one or two experts!

**Why this happens:**
```
Initially: All experts equally good (random init)
Epoch 1: Expert 3 happens to perform slightly better
Epoch 2: Router sends more tokens to Expert 3
         Expert 3 gets more gradients, learns faster
Epoch 3: Router sends even MORE tokens to Expert 3
         Other experts get few/no tokens, don't learn
Result: Expert 3 handles 90% of tokens, others wasted!
```

**Real example from research:**
```
Without load balancing:
- Expert 0: 82% of tokens
- Expert 1: 12% of tokens
- Expert 2-7: 6% of tokens total
→ Effectively a 1-expert model!

With load balancing:
- All experts: 12-13% of tokens each
→ Actually using all experts!
```

### Auxiliary Load Balancing Loss

**Solution:** Add a penalty for unbalanced expert usage.

**The loss function:**
```
Load Balancing Loss = N × Σ(f_i × P_i)

Where:
- N = number of experts
- f_i = fraction of tokens sent to expert i
- P_i = average router probability for expert i
```

**Intuition:**
- If expert i gets many tokens (high f_i) AND has high probabilities (high P_i), the product is large
- This penalizes experts that are "too popular"
- Encourages router to distribute load evenly

**Example calculation:**
```csharp
// Suppose 8 experts, 100 tokens
int[] tokenCounts = { 50, 10, 10, 10, 5, 5, 5, 5 }; // Unbalanced!
double[] avgProbs = { 0.6, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05 };

// Compute f_i (fraction)
double[] f = tokenCounts.Select(c => c / 100.0).ToArray();
// f = [0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]

// Compute load balancing loss
double loss = 0;
for (int i = 0; i < 8; i++)
{
    loss += f[i] * avgProbs[i];
}
loss *= 8; // Multiply by N

// loss = 8 × (0.5×0.6 + 0.1×0.05 + ...) = 8 × 0.34 = 2.72

// If perfectly balanced: f_i = 0.125, P_i = 0.125 for all i
// Perfect loss = 8 × (0.125 × 0.125) × 8 = 1.0

// Penalty for imbalance: 2.72 - 1.0 = 1.72
```

**For Beginners:**
- Loss is LOW when experts are used equally
- Loss is HIGH when some experts dominate
- During training, we add this to the main loss to encourage balance

---

## Implementation Guide

### Phase 1: Core MoE Components

#### Step 1: Expert Class

**File:** `src/NeuralNetworks/Layers/Experts/Expert.cs`

```csharp
using AiDotNet.Mathematics;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Layers.Experts;

/// <summary>
/// Represents a single expert in a Mixture-of-Experts architecture.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> An expert is a specialized sub-network.
///
/// Think of it as one member of a team of specialists. Each expert is a
/// complete feed-forward network that can process inputs independently.
/// In MoE, you have many experts, but only a few are activated per input.
/// </para>
/// </remarks>
public class Expert<T> : LayerBase<T>
{
    private readonly DenseLayer<T> _upProjection;
    private readonly DenseLayer<T> _downProjection;
    private readonly IActivationFunction<T> _activation;
    private readonly int _inputSize;
    private readonly int _hiddenSize;

    /// <summary>
    /// Initializes a new expert network.
    /// </summary>
    /// <param name="inputSize">Input dimension (e.g., 512)</param>
    /// <param name="hiddenSize">Hidden dimension (e.g., 2048).
    /// <b>Why 4× expansion?</b> Standard practice from Transformer FFN,
    /// provides enough capacity for complex transformations.</param>
    public Expert(int inputSize, int? hiddenSize = null)
        : base(new[] { inputSize }, new[] { inputSize })
    {
        if (inputSize <= 0)
            throw new ArgumentException("Input size must be positive", nameof(inputSize));

        _inputSize = inputSize;
        _hiddenSize = hiddenSize ?? (inputSize * 4);

        // Two-layer FFN with ReLU
        _upProjection = new DenseLayer<T>(inputSize, _hiddenSize);
        _downProjection = new DenseLayer<T>(_hiddenSize, inputSize);
        _activation = new ReLUActivation<T>();
    }

    public override bool SupportsTraining => true;

    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Standard feed-forward: input → expand → activate → compress
        var expanded = _upProjection.Forward(input);
        var activated = expanded.Transform((val, _) => _activation.Activate(val));
        var output = _downProjection.Forward(activated);
        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Backprop through the FFN
        var grad = _downProjection.Backward(outputGradient);
        grad = grad.Transform((val, _) => _activation.Derivative(val));
        grad = _upProjection.Backward(grad);
        return grad;
    }

    public override void UpdateParameters(T learningRate)
    {
        _upProjection.UpdateParameters(learningRate);
        _downProjection.UpdateParameters(learningRate);
    }

    public override Vector<T> GetParameters()
    {
        var params1 = _upProjection.GetParameters();
        var params2 = _downProjection.GetParameters();

        Vector<T> allParams = new Vector<T>(params1.Length + params2.Length);
        for (int i = 0; i < params1.Length; i++)
            allParams[i] = params1[i];
        for (int i = 0; i < params2.Length; i++)
            allParams[params1.Length + i] = params2[i];

        return allParams;
    }

    public override void ResetState()
    {
        _upProjection.ResetState();
        _downProjection.ResetState();
    }
}
```

#### Step 2: MixtureOfExpertsLayer

**File:** `src/NeuralNetworks/Layers/MixtureOfExpertsLayer.cs`

```csharp
using AiDotNet.Mathematics;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.Experts;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Interface for layers that contribute auxiliary losses (like load balancing).
/// </summary>
public interface IAuxiliaryLossLayer<T>
{
    /// <summary>
    /// Gets the auxiliary loss for this layer (e.g., load balancing loss).
    /// </summary>
    Tensor<T> GetAuxiliaryLoss();
}

/// <summary>
/// Implements a Mixture-of-Experts layer with top-k routing.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This layer contains multiple expert networks and a router.
///
/// Think of it like a smart switchboard:
/// - Incoming data arrives
/// - Router decides which experts should handle it
/// - Only the selected experts process the data
/// - Results are combined based on router's confidence
///
/// This allows for massive model capacity without massive computation cost!
/// </para>
/// </remarks>
public class MixtureOfExpertsLayer<T> : LayerBase<T>, IAuxiliaryLossLayer<T>
{
    private readonly int _inputSize;
    private readonly int _numExperts;
    private readonly int _topK;

    private readonly List<Expert<T>> _experts;
    private readonly DenseLayer<T> _router;

    // For computing load balancing loss
    private Tensor<T>? _lastRouterProbs;
    private int[]? _lastExpertAssignments;

    /// <summary>
    /// Initializes a new Mixture-of-Experts layer.
    /// </summary>
    /// <param name="inputSize">Input dimension (e.g., 512)</param>
    /// <param name="outputSize">Output dimension (usually same as input)</param>
    /// <param name="numExperts">Number of expert networks (e.g., 8).
    /// <b>Why 8?</b> Common choice balancing capacity vs. complexity.
    /// More experts = more capacity but more overhead.</param>
    /// <param name="topK">Number of experts to activate per token (e.g., 2).
    /// <b>Why 2?</b> Good balance - uses specialized knowledge without too much compute.
    /// Research shows top-2 often performs best.</param>
    public MixtureOfExpertsLayer(
        int inputSize,
        int outputSize,
        int numExperts = 8,
        int topK = 2)
        : base(new[] { inputSize }, new[] { outputSize })
    {
        if (inputSize <= 0)
            throw new ArgumentException("Input size must be positive", nameof(inputSize));
        if (outputSize <= 0)
            throw new ArgumentException("Output size must be positive", nameof(outputSize));
        if (numExperts <= 0)
            throw new ArgumentException("Number of experts must be positive", nameof(numExperts));
        if (topK <= 0 || topK > numExperts)
            throw new ArgumentException($"Top-k must be between 1 and {numExperts}", nameof(topK));

        _inputSize = inputSize;
        _numExperts = numExperts;
        _topK = topK;

        // Initialize experts
        _experts = new List<Expert<T>>(numExperts);
        for (int i = 0; i < numExperts; i++)
        {
            _experts.Add(new Expert<T>(inputSize, hiddenSize: inputSize * 4));
        }

        // Initialize router (projects to num_experts logits)
        _router = new DenseLayer<T>(inputSize, numExperts);
    }

    public override bool SupportsTraining => true;

    public override Tensor<T> Forward(Tensor<T> input)
    {
        // input: (batch, seq_len, input_size)
        int batch = input.Shape[0];
        int seqLen = input.Shape[1];
        int totalTokens = batch * seqLen;

        // 1. Router: compute expert probabilities for each token
        Tensor<T> routerLogits = _router.Forward(input); // (batch, seq_len, num_experts)
        Tensor<T> routerProbs = Softmax(routerLogits, axis: 2);

        _lastRouterProbs = routerProbs; // Save for load balancing loss

        // 2. For each token, select top-k experts
        var expertAssignments = new List<(int tokenIdx, int expertIdx, T weight)>();
        var expertToTokens = new Dictionary<int, List<(int tokenIdx, T weight)>>();

        for (int e = 0; e < _numExperts; e++)
            expertToTokens[e] = new List<(int tokenIdx, T weight)>();

        int tokenIdx = 0;
        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                // Get probabilities for this token
                T[] probs = new T[_numExperts];
                for (int e = 0; e < _numExperts; e++)
                    probs[e] = routerProbs[b, t, e];

                // Select top-k
                var topK = SelectTopK(probs, _topK);

                // Renormalize weights
                T sumWeights = NumOps.Zero;
                foreach (var w in topK.weights)
                    sumWeights = NumOps.Add(sumWeights, w);

                for (int k = 0; k < _topK; k++)
                {
                    T normalizedWeight = NumOps.Divide(topK.weights[k], sumWeights);
                    int expertIdx = topK.indices[k];

                    expertToTokens[expertIdx].Add((tokenIdx, normalizedWeight));
                    expertAssignments.Add((tokenIdx, expertIdx, normalizedWeight));
                }

                tokenIdx++;
            }
        }

        // Save for load balancing loss
        _lastExpertAssignments = expertAssignments
            .Select(a => a.expertIdx)
            .ToArray();

        // 3. Process tokens through experts (batched by expert)
        Tensor<T> output = new Tensor<T>(input.Shape);

        // Flatten input for easier indexing: (total_tokens, input_size)
        Tensor<T> flatInput = input.Reshape(totalTokens, _inputSize);
        Tensor<T> flatOutput = new Tensor<T>(new[] { totalTokens, _inputSize });

        foreach (var (expertIdx, tokens) in expertToTokens)
        {
            if (tokens.Count == 0)
                continue; // Expert not used for this batch

            // Gather inputs for this expert
            Tensor<T> expertInput = new Tensor<T>(new[] { tokens.Count, _inputSize });
            for (int i = 0; i < tokens.Count; i++)
            {
                int tIdx = tokens[i].tokenIdx;
                for (int d = 0; d < _inputSize; d++)
                    expertInput[i, d] = flatInput[tIdx, d];
            }

            // Process through expert
            Tensor<T> expertOutput = _experts[expertIdx].Forward(expertInput);

            // Scatter outputs back (weighted)
            for (int i = 0; i < tokens.Count; i++)
            {
                int tIdx = tokens[i].tokenIdx;
                T weight = tokens[i].weight;

                for (int d = 0; d < _inputSize; d++)
                {
                    T weightedVal = NumOps.Multiply(expertOutput[i, d], weight);
                    flatOutput[tIdx, d] = NumOps.Add(flatOutput[tIdx, d], weightedVal);
                }
            }
        }

        // Reshape back to original dimensions
        output = flatOutput.Reshape(batch, seqLen, _inputSize);

        return output;
    }

    /// <summary>
    /// Computes the load balancing auxiliary loss.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This loss encourages balanced expert usage.
    ///
    /// Without this, all tokens might go to just 1-2 experts, wasting the others.
    /// The loss penalizes imbalanced routing, encouraging the router to spread
    /// tokens evenly across all experts.
    ///
    /// Formula: N × Σ(f_i × P_i)
    /// Where f_i = fraction of tokens to expert i
    ///       P_i = average router probability for expert i
    /// </para>
    /// </remarks>
    public Tensor<T> GetAuxiliaryLoss()
    {
        if (_lastRouterProbs == null || _lastExpertAssignments == null)
            return Tensor<T>.CreateDefault(new[] { 1 }, NumOps.Zero);

        int batch = _lastRouterProbs.Shape[0];
        int seqLen = _lastRouterProbs.Shape[1];
        int totalTokens = batch * seqLen;

        // Compute f_i: fraction of tokens assigned to each expert
        Vector<T> f = new Vector<T>(_numExperts);
        foreach (int expertIdx in _lastExpertAssignments)
        {
            f[expertIdx] = NumOps.Add(f[expertIdx], NumOps.One);
        }
        for (int i = 0; i < _numExperts; i++)
        {
            f[i] = NumOps.Divide(f[i], NumOps.FromDouble(totalTokens));
        }

        // Compute P_i: average router probability for each expert
        Vector<T> P = new Vector<T>(_numExperts);
        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                for (int e = 0; e < _numExperts; e++)
                {
                    P[e] = NumOps.Add(P[e], _lastRouterProbs[b, t, e]);
                }
            }
        }
        for (int i = 0; i < _numExperts; i++)
        {
            P[i] = NumOps.Divide(P[i], NumOps.FromDouble(totalTokens));
        }

        // Compute load balancing loss: N × Σ(f_i × P_i)
        T loss = NumOps.Zero;
        for (int i = 0; i < _numExperts; i++)
        {
            loss = NumOps.Add(loss, NumOps.Multiply(f[i], P[i]));
        }
        loss = NumOps.Multiply(loss, NumOps.FromDouble(_numExperts));

        return Tensor<T>.CreateDefault(new[] { 1 }, loss);
    }

    private (int[] indices, T[] weights) SelectTopK(T[] values, int k)
    {
        // Create indices
        var indexedValues = values
            .Select((val, idx) => (val, idx))
            .OrderByDescending(x => Convert.ToDouble(x.val))
            .Take(k)
            .ToArray();

        return (
            indexedValues.Select(x => x.idx).ToArray(),
            indexedValues.Select(x => x.val).ToArray()
        );
    }

    private Tensor<T> Softmax(Tensor<T> input, int axis)
    {
        // Compute softmax along specified axis
        // For simplicity, assuming axis = 2 (last dimension)
        Tensor<T> output = new Tensor<T>(input.Shape);

        int batch = input.Shape[0];
        int seqLen = input.Shape[1];
        int dim = input.Shape[2];

        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                // Find max for numerical stability
                T maxVal = input[b, t, 0];
                for (int d = 1; d < dim; d++)
                {
                    if (NumOps.GreaterThan(input[b, t, d], maxVal))
                        maxVal = input[b, t, d];
                }

                // Compute exp and sum
                T sum = NumOps.Zero;
                T[] exp_vals = new T[dim];
                for (int d = 0; d < dim; d++)
                {
                    T shifted = NumOps.Subtract(input[b, t, d], maxVal);
                    exp_vals[d] = NumOps.Exp(shifted);
                    sum = NumOps.Add(sum, exp_vals[d]);
                }

                // Normalize
                for (int d = 0; d < dim; d++)
                {
                    output[b, t, d] = NumOps.Divide(exp_vals[d], sum);
                }
            }
        }

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // TODO: Implement backward pass
        throw new NotImplementedException("Backward pass for MoE coming in Phase 2");
    }

    public override void UpdateParameters(T learningRate)
    {
        _router.UpdateParameters(learningRate);
        foreach (var expert in _experts)
            expert.UpdateParameters(learningRate);
    }

    public override Vector<T> GetParameters()
    {
        var allParams = new List<Vector<T>> { _router.GetParameters() };
        foreach (var expert in _experts)
            allParams.Add(expert.GetParameters());

        int totalLength = allParams.Sum(p => p.Length);
        Vector<T> result = new Vector<T>(totalLength);
        int offset = 0;

        foreach (var p in allParams)
        {
            for (int i = 0; i < p.Length; i++)
                result[offset++] = p[i];
        }

        return result;
    }

    public override void ResetState()
    {
        _router.ResetState();
        foreach (var expert in _experts)
            expert.ResetState();
    }
}
```

---

## Testing Strategy

### Test 1: Expert Specialization

```csharp
[Fact]
public void MoE_ForwardPass_ProducesCorrectShape()
{
    // Arrange
    int inputSize = 64;
    int numExperts = 4;
    int topK = 2;
    int batch = 2;
    int seqLen = 5;

    var moe = new MixtureOfExpertsLayer<double>(
        inputSize: inputSize,
        outputSize: inputSize,
        numExperts: numExperts,
        topK: topK);

    var input = new Tensor<double>(new[] { batch, seqLen, inputSize });
    FillRandom(input);

    // Act
    var output = moe.Forward(input);

    // Assert
    Assert.Equal(new[] { batch, seqLen, inputSize }, output.Shape);

    // Check for NaN/Inf
    foreach (var val in output.ToArray())
    {
        Assert.False(double.IsNaN(val));
        Assert.False(double.IsInfinity(val));
    }
}
```

### Test 2: Load Balancing Loss

```csharp
[Fact]
public void MoE_LoadBalancingLoss_IncreasesWithImbalance()
{
    // Arrange
    int inputSize = 32;
    int numExperts = 4;
    var moe = new MixtureOfExpertsLayer<double>(
        inputSize, inputSize, numExperts, topK: 1);

    // Create two batches:
    // 1. Balanced: diverse inputs should spread across experts
    var balancedInput = CreateDiverseInput(batch: 1, seqLen: 100, inputSize);

    // 2. Unbalanced: similar inputs should cluster to few experts
    var unbalancedInput = CreateSimilarInput(batch: 1, seqLen: 100, inputSize);

    // Act
    var _ = moe.Forward(balancedInput);
    var balancedLoss = moe.GetAuxiliaryLoss()[0];

    moe.ResetState(); // Clear routing history

    var __ = moe.Forward(unbalancedInput);
    var unbalancedLoss = moe.GetAuxiliaryLoss()[0];

    // Assert: Unbalanced should have higher loss
    Assert.True(unbalancedLoss > balancedLoss,
        $"Expected unbalanced loss ({unbalancedLoss}) > balanced loss ({balancedLoss})");
}
```

### Test 3: End-to-End Training with Auxiliary Loss

```csharp
[Fact]
public void MoE_Training_BalancesExpertUsage()
{
    // Arrange
    int inputSize = 32;
    int numExperts = 4;
    var moe = new MixtureOfExpertsLayer<double>(inputSize, inputSize, numExperts, topK: 2);

    var input = CreateRandomInput(batch: 4, seqLen: 20, inputSize);
    var target = CreateRandomTarget(batch: 4, seqLen: 20, inputSize);

    double alpha = 0.01; // Load balancing coefficient

    // Track expert usage over epochs
    var expertUsageBefore = CountExpertUsage(moe, input);

    // Act: Train for multiple epochs with auxiliary loss
    for (int epoch = 0; epoch < 50; epoch++)
    {
        var output = moe.Forward(input);

        // Compute main loss (MSE)
        var mainLoss = ComputeMSE(output, target);

        // Compute auxiliary loss
        var auxLoss = moe.GetAuxiliaryLoss();

        // Total loss: main + alpha × auxiliary
        var totalLoss = mainLoss + alpha * auxLoss[0];

        // Backprop and update (simplified)
        // In real implementation, you'd backprop through both losses
        // ...
    }

    var expertUsageAfter = CountExpertUsage(moe, input);

    // Assert: Expert usage should be more balanced after training
    double varianceBefore = ComputeVariance(expertUsageBefore);
    double varianceAfter = ComputeVariance(expertUsageAfter);

    Assert.True(varianceAfter < varianceBefore,
        "Expert usage should become more balanced with load balancing loss");
}
```

---

## Common Pitfalls

### 1. Router Collapse

**Problem:** All tokens routed to one expert.

**Solution:** Use load balancing loss with appropriate weight.

```csharp
// BAD: No load balancing
var loss = mainLoss;

// GOOD: Include auxiliary loss
var auxLoss = moeLayer.GetAuxiliaryLoss();
var loss = mainLoss + 0.01 * auxLoss; // 0.01 is typical alpha
```

### 2. Gradient Starvation

**Problem:** Experts that receive few tokens get few gradient updates.

**Solution:** Ensure minimum token assignment or use expert dropout.

```csharp
// Add noise to router logits to prevent deterministic routing
var logits = routerLayer.Forward(input);
if (IsTrainingMode)
{
    var noise = SampleGaussianNoise(logits.Shape);
    logits = logits.Add(noise.Multiply(0.01)); // Small noise
}
```

### 3. Inefficient Token Batching

**Problem:** Processing one token at a time through each expert.

**Solution:** Batch tokens by expert.

```csharp
// BAD: Process tokens individually
foreach (var token in tokens)
{
    var expertIdx = router.Select(token);
    var output = experts[expertIdx].Forward(token); // Slow!
}

// GOOD: Batch by expert
var expertBatches = GroupTokensByExpert(tokens, router);
foreach (var (expertIdx, batch) in expertBatches)
{
    var outputs = experts[expertIdx].Forward(batch); // Fast!
}
```

### 4. Numerical Instability in Softmax

**Problem:** Overflow in exp() for large logits.

**Solution:** Use log-sum-exp trick.

```csharp
// BAD
T exp_sum = logits.Sum(x => NumOps.Exp(x)); // Can overflow!

// GOOD
T max_logit = logits.Max();
T exp_sum = logits.Sum(x => NumOps.Exp(NumOps.Subtract(x, max_logit)));
```

### 5. Forgetting to Use Auxiliary Loss

**Problem:** Implementing load balancing but not including it in training.

**Critical:** Update your training loop!

```csharp
// In training loop, check for auxiliary loss layers
public void Train(NeuralNetwork model, Dataset data)
{
    foreach (var batch in data)
    {
        var output = model.Forward(batch.input);
        var mainLoss = LossFunction(output, batch.target);

        // Collect auxiliary losses from all MoE layers
        var auxLoss = 0.0;
        foreach (var layer in model.Layers)
        {
            if (layer is IAuxiliaryLossLayer<double> auxLayer)
            {
                auxLoss += auxLayer.GetAuxiliaryLoss()[0];
            }
        }

        // Total loss
        var totalLoss = mainLoss + alpha * auxLoss;

        // Backprop with total loss
        model.Backward(totalLoss);
        model.UpdateParameters(learningRate);
    }
}
```

---

## Summary

**What you've learned:**

1. **MoE architecture** enables massive model capacity with controlled compute through sparse activation
2. **Top-k routing** selects specialized experts for each input, enabling automatic specialization
3. **Load balancing** prevents expert collapse and ensures all experts contribute to learning
4. **Auxiliary loss** guides the router to distribute tokens evenly across experts
5. **Efficient implementation** requires careful batching of tokens by expert

**Key benefits:**
- Scale to trillions of parameters with manageable training cost
- Automatic specialization of experts for different input types
- Faster inference than equivalent dense models
- State-of-the-art performance on language tasks (GPT-4, Switch Transformer)

**Next steps:**
- Implement backward pass for end-to-end training
- Add expert dropout for better generalization
- Implement capacity factor to limit tokens per expert
- Benchmark against dense baseline

**Resources:**
- Switch Transformer paper: https://arxiv.org/abs/2101.03961
- GShard paper: https://arxiv.org/abs/2006.16668
- Expert Choice Routing: https://arxiv.org/abs/2202.09368
- GPT-4 architecture (rumored to use MoE): https://the-decoder.com/gpt-4-architecture-datasets-costs-and-more-leaked/

# Issue #276: RWKV-Style RNN as Transformer Alternative - Implementation Guide

## For Junior Developers: Complete Implementation Tutorial

### Table of Contents
1. [Understanding RWKV](#understanding-rwkv)
2. [Why RWKV Beats Traditional RNNs and Rivals Transformers](#advantages)
3. [Architecture Deep Dive](#architecture)
4. [Parallel vs Sequential Modes](#dual-mode-operation)
5. [Implementation Guide](#implementation-guide)
6. [Testing Strategy](#testing-strategy)
7. [Common Pitfalls](#common-pitfalls)

---

## Understanding RWKV

### What is RWKV?

**For Beginners:** RWKV (Receptance Weighted Key Value) is a revolutionary architecture that combines the best of two worlds:
- **Training**: As fast as Transformers (fully parallelizable)
- **Inference**: As efficient as RNNs (constant memory, linear time)

**The magic:** RWKV can process sequences in two different ways:
1. **Parallel mode** (training): Process entire sequence at once, like a Transformer
2. **Sequential mode** (inference): Process one token at a time, like an RNN

**Real-world analogy:**
- Traditional RNN: Reading a book word-by-word, can only move forward
- Transformer: Photocopying the entire book and analyzing all words simultaneously
- RWKV: Can do both! Train like a copier (parallel), but run like a reader (sequential)

**Key innovation:** A clever mathematical reformulation allows the same operation to be computed either way!

---

## Advantages

### Why Choose RWKV?

#### 1. Best of Both Worlds

**RNN Problems:**
- Slow to train (sequential processing)
- Limited context (vanishing gradients)
- But: Fast inference, constant memory

**Transformer Problems:**
- Quadratic memory (attention matrix)
- Expensive for long sequences
- But: Fast parallel training

**RWKV Solution:**
- ✅ Fast parallel training (like Transformers)
- ✅ Linear memory (like RNNs)
- ✅ Constant inference memory (like RNNs)
- ✅ Long context handling (better than vanilla RNNs)

#### 2. Memory Efficiency Comparison

**Sequence of 10,000 tokens:**

```csharp
// Transformer (self-attention)
// Attention matrix: (10000 × 10000) = 100 million values!
Matrix<T> attention = new Matrix<T>(10000, 10000); // ~400MB for float32

// RWKV (sequential inference)
// State vector: constant size regardless of sequence length!
Vector<T> state = new Vector<T>(1024); // ~4KB for float32
// 100,000× less memory!
```

#### 3. Inference Speed

**Generating 100 new tokens:**

**Transformer:**
```csharp
// Must recompute attention over ALL previous tokens each time
for (int i = 0; i < 100; i++)
{
    // Attention matrix grows: (i × i)
    // Token 1: 1×1, Token 2: 2×2, ..., Token 100: 100×100
    // Total: O(N²) operations
}
```

**RWKV:**
```csharp
// Just update fixed-size state
Vector<T> state = initialState;
for (int i = 0; i < 100; i++)
{
    (output, state) = ProcessToken(input[i], state);
    // Same cost every iteration: O(D²) where D is state size
}
// Total: O(N) operations - 100× faster!
```

#### 4. Training Speed

**RWKV vs RNN:**
```csharp
// Traditional RNN: Must process sequentially
Vector<T> hidden = initialState;
for (int t = 0; t < sequenceLength; t++)
{
    hidden = RNNStep(hidden, input[t]); // Can't parallelize!
}
// Time: O(L) serial steps

// RWKV: Can process all at once!
Matrix<T> allInputs = GetBatch(sequenceLength); // All tokens
Matrix<T> allOutputs = RWKVParallel(allInputs);  // One parallel operation!
// Time: O(1) parallel steps (with enough processors)
```

---

## Architecture

### The RWKV Block Structure

RWKV consists of alternating blocks of two types:
1. **Time-Mixing**: Processes information across the time dimension (sequence)
2. **Channel-Mixing**: Processes information across the feature dimension

```
Input Embedding
    ↓
[Time-Mixing + LayerNorm + Residual]
    ↓
[Channel-Mixing + LayerNorm + Residual]
    ↓
[Time-Mixing + LayerNorm + Residual]
    ↓
[Channel-Mixing + LayerNorm + Residual]
    ↓
    ... (repeat for N layers)
    ↓
Output Linear Layer → Logits
```

**For Beginners:**
- **Time-Mixing**: Like attention, but recurrent - understands relationships over time
- **Channel-Mixing**: Like a feed-forward network - transforms features
- **Residual connections**: Add input to output, helping gradients flow during training

---

### Time-Mixing Block: The Heart of RWKV

This is where RWKV's magic happens - the ability to mix information across time steps.

#### Core Idea: Weighted Key-Value Memory

**Traditional Attention:**
```
For each position, compute attention weights to ALL other positions
Output = weighted sum of ALL values
```

**RWKV Time-Mixing:**
```
Maintain a running weighted average of key-value pairs
Update: blend previous state with current input
Output: read from updated state
```

#### The WKV (Weighted Key-Value) Operation

**Mathematical formulation:**

```
// At each time step t:
k_t = W_k × (μ_k ⊙ x_{t-1} + (1 - μ_k) ⊙ x_t)   [Key]
v_t = W_v × (μ_v ⊙ x_{t-1} + (1 - μ_v) ⊙ x_t)   [Value]
r_t = W_r × (μ_r ⊙ x_{t-1} + (1 - μ_r) ⊙ x_t)   [Receptance - like a gate]

// WKV computation:
wkv_t = (Σ_{i=1}^{t} exp(w_{i→t} + k_i) × v_i) / (Σ_{i=1}^{t} exp(w_{i→t} + k_i))

// Output:
output_t = W_o × (sigmoid(r_t) ⊙ wkv_t)
```

Where:
- `μ` (mu): Learnable interpolation between current and previous input
- `w`: Time-decay weights (exponential decay)
- `⊙`: Element-wise multiplication

**For Beginners:**
- Think of `k` and `v` as key-value pairs in a memory system
- `r` (receptance) controls how much to read from memory
- `w` makes recent information more important (exponential decay)
- The "weighted average" part is computed efficiently in both parallel and sequential modes!

#### Dual Implementation: Parallel vs Sequential

**The Key Insight:** The WKV operation can be reformulated!

**Parallel Mode (Training):**
```csharp
public static Tensor<T> WKV_Parallel<T>(
    Tensor<T> k,  // Keys: (batch, seq_len, d_model)
    Tensor<T> v,  // Values: (batch, seq_len, d_model)
    Tensor<T> w)  // Time decay: (d_model,)
{
    int batch = k.Shape[0];
    int seqLen = k.Shape[1];
    int dModel = k.Shape[2];

    Tensor<T> wkv = new Tensor<T>(new[] { batch, seqLen, dModel });

    // Process all time steps in parallel
    for (int b = 0; b < batch; b++)
    {
        for (int t = 0; t < seqLen; t++)
        {
            // For each position t, compute weighted sum over all i ≤ t
            for (int d = 0; d < dModel; d++)
            {
                T numerator = NumOps.Zero;
                T denominator = NumOps.Zero;

                for (int i = 0; i <= t; i++)
                {
                    // Time decay: w^(t-i)
                    T decay = NumOps.Exp(NumOps.Multiply(
                        w[d],
                        NumOps.FromDouble(t - i)));

                    T weight = NumOps.Exp(NumOps.Add(
                        NumOps.Log(decay),
                        k[b, i, d]));

                    numerator = NumOps.Add(numerator,
                        NumOps.Multiply(weight, v[b, i, d]));
                    denominator = NumOps.Add(denominator, weight);
                }

                wkv[b, t, d] = NumOps.Divide(numerator, denominator);
            }
        }
    }

    return wkv;
}
```

**Sequential Mode (Inference):**
```csharp
public static (Vector<T> output, WKVState<T> newState) WKV_Sequential<T>(
    Vector<T> k_t,        // Current key: (d_model,)
    Vector<T> v_t,        // Current value: (d_model,)
    Vector<T> w,          // Time decay: (d_model,)
    WKVState<T> state)    // Previous state
{
    int dModel = k_t.Length;
    Vector<T> output = new Vector<T>(dModel);

    // State contains running numerator and denominator
    Vector<T> num = state.Numerator;
    Vector<T> den = state.Denominator;

    for (int d = 0; d < dModel; d++)
    {
        // Decay previous accumulator
        T decay = NumOps.Exp(w[d]);
        num[d] = NumOps.Multiply(num[d], decay);
        den[d] = NumOps.Multiply(den[d], decay);

        // Add current contribution
        T weight = NumOps.Exp(k_t[d]);
        num[d] = NumOps.Add(num[d], NumOps.Multiply(weight, v_t[d]));
        den[d] = NumOps.Add(den[d], weight);

        // Output
        output[d] = NumOps.Divide(num[d], den[d]);
    }

    return (output, new WKVState<T> { Numerator = num, Denominator = den });
}
```

**Critical:** These two implementations MUST produce identical results for the same input!

---

### Channel-Mixing Block

**Much simpler than Time-Mixing!**

This is essentially a position-wise feed-forward network with a gating mechanism.

```csharp
public class ChannelMixing<T> : LayerBase<T>
{
    private readonly DenseLayer<T> _W_k;
    private readonly DenseLayer<T> _W_v;
    private readonly DenseLayer<T> _W_r;
    private readonly Vector<T> _mu_k;
    private readonly Vector<T> _mu_v;
    private readonly Vector<T> _mu_r;

    public override Tensor<T> Forward(Tensor<T> x)
    {
        // x: (batch, seq_len, d_model)
        // x_prev: previous time step (or zero for t=0)

        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int dModel = x.Shape[2];

        Tensor<T> output = new Tensor<T>(x.Shape);

        for (int b = 0; b < batch; b++)
        {
            Vector<T> x_prev = new Vector<T>(dModel); // Initialize to zero

            for (int t = 0; t < seqLen; t++)
            {
                Vector<T> x_t = GetVector(x, b, t);

                // Interpolate between current and previous
                Vector<T> x_k = Interpolate(x_prev, x_t, _mu_k);
                Vector<T> x_v = Interpolate(x_prev, x_t, _mu_v);
                Vector<T> x_r = Interpolate(x_prev, x_t, _mu_r);

                // Apply transformations
                Vector<T> k = _W_k.Forward(x_k);
                Vector<T> v = _W_v.Forward(x_v);
                Vector<T> r = _W_r.Forward(x_r);

                // Squared ReLU activation on v
                v = v.Transform(val => {
                    T relu = NumOps.Max(NumOps.Zero, val);
                    return NumOps.Multiply(relu, relu);
                });

                // Sigmoid gate on r
                r = r.Transform(val => {
                    T exp_neg = NumOps.Exp(NumOps.Negate(val));
                    return NumOps.Divide(NumOps.One,
                        NumOps.Add(NumOps.One, exp_neg));
                });

                // Output: r ⊙ v
                Vector<T> out_t = ElementwiseMultiply(r, v);
                SetVector(output, b, t, out_t);

                // Update x_prev for next iteration
                x_prev = x_t;
            }
        }

        return output;
    }

    private Vector<T> Interpolate(Vector<T> prev, Vector<T> curr, Vector<T> mu)
    {
        // result[i] = mu[i] × prev[i] + (1 - mu[i]) × curr[i]
        Vector<T> result = new Vector<T>(prev.Length);
        for (int i = 0; i < prev.Length; i++)
        {
            T term1 = NumOps.Multiply(mu[i], prev[i]);
            T one_minus_mu = NumOps.Subtract(NumOps.One, mu[i]);
            T term2 = NumOps.Multiply(one_minus_mu, curr[i]);
            result[i] = NumOps.Add(term1, term2);
        }
        return result;
    }
}
```

**For Beginners:**
- Interpolation (μ): Blends current input with previous, allowing the model to see temporal patterns
- Squared ReLU: A variant of ReLU that emphasizes larger values
- Sigmoid gate: Controls information flow (like LSTM gates)

---

## Dual Mode Operation

### Why Both Modes Are Essential

**Training (Parallel Mode):**
- Process entire batch of sequences simultaneously
- Utilize GPU parallelism efficiently
- Compute gradients for all time steps at once
- **Must use parallel implementation for speed**

**Inference (Sequential Mode):**
- Generate text one token at a time
- Maintain constant memory regardless of context length
- Update state incrementally
- **Must use sequential implementation for efficiency**

### The Critical Test: Equivalence

**The most important test for RWKV:**

```csharp
[Fact]
public void TimeMixing_ParallelAndSequentialMode_ProduceIdenticalResults()
{
    // Arrange
    int dModel = 64;
    int seqLen = 5;
    var timeMixing = new TimeMixing<double>(dModel);

    // Create random input sequence
    var input = CreateRandomTensor(1, seqLen, dModel);

    // Act: Parallel mode (process entire sequence)
    var outputParallel = timeMixing.Forward(input, mode: ProcessingMode.Parallel);

    // Act: Sequential mode (process token by token)
    var outputSequential = new Tensor<double>(input.Shape);
    var state = timeMixing.InitializeState();

    for (int t = 0; t < seqLen; t++)
    {
        var input_t = input.Slice(1, t, t + 1); // Get token at position t
        var (output_t, newState) = timeMixing.ForwardSequential(input_t, state);
        outputSequential.SetSlice(1, t, output_t);
        state = newState;
    }

    // Assert: Outputs must be numerically identical!
    for (int t = 0; t < seqLen; t++)
    {
        for (int d = 0; d < dModel; d++)
        {
            double diff = Math.Abs(
                outputParallel[0, t, d] - outputSequential[0, t, d]);

            Assert.True(diff < 1e-5,
                $"Mismatch at t={t}, d={d}: " +
                $"parallel={outputParallel[0, t, d]}, " +
                $"sequential={outputSequential[0, t, d]}");
        }
    }
}
```

**This test is CRITICAL because:**
1. It verifies mathematical correctness
2. Ensures training and inference use the same model
3. Catches numerical stability issues
4. Validates the core innovation of RWKV

---

## Implementation Guide

### Phase 1: Time-Mixing Block

#### Step 1: Define the WKV State

**File:** `src/NeuralNetworks/Layers/RWKV/WKVState.cs`

```csharp
namespace AiDotNet.NeuralNetworks.Layers.RWKV;

/// <summary>
/// Maintains the hidden state for WKV (Weighted Key-Value) computation in sequential mode.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is the "memory" that RWKV maintains between tokens.
///
/// Unlike Transformers that keep a growing cache of all previous tokens,
/// RWKV just keeps two vectors (numerator and denominator) that summarize
/// the entire history. This is why RWKV has constant memory!
/// </para>
/// </remarks>
public class WKVState<T>
{
    /// <summary>
    /// Running weighted sum of values (numerator in WKV formula).
    /// </summary>
    public Vector<T> Numerator { get; set; }

    /// <summary>
    /// Running sum of weights (denominator in WKV formula).
    /// </summary>
    public Vector<T> Denominator { get; set; }

    /// <summary>
    /// Previous input (for interpolation in next step).
    /// </summary>
    public Vector<T> PreviousInput { get; set; }

    public WKVState(int dimension)
    {
        Numerator = new Vector<T>(dimension);
        Denominator = new Vector<T>(dimension);
        PreviousInput = new Vector<T>(dimension);
    }
}
```

#### Step 2: Implement Time-Mixing Layer

**File:** `src/NeuralNetworks/Layers/RWKV/TimeMixing.cs`

```csharp
using AiDotNet.Mathematics;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Layers.RWKV;

/// <summary>
/// Implements the Time-Mixing block of RWKV architecture.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Time-Mixing is RWKV's replacement for attention.
///
/// It processes information across the sequence (over time) using a clever
/// weighted key-value mechanism that can run in two modes:
/// - Parallel: Process entire sequence at once (training)
/// - Sequential: Process one token at a time (inference)
///
/// Both modes produce identical outputs - this is the key innovation!
/// </para>
/// </remarks>
public class TimeMixing<T> : LayerBase<T>
{
    private readonly int _dModel;
    private readonly DenseLayer<T> _W_k;
    private readonly DenseLayer<T> _W_v;
    private readonly DenseLayer<T> _W_r;
    private readonly DenseLayer<T> _W_o;

    // Learnable interpolation parameters
    private Vector<T> _mu_k;  // Mix ratio for keys
    private Vector<T> _mu_v;  // Mix ratio for values
    private Vector<T> _mu_r;  // Mix ratio for receptance

    // Time decay weights
    private Vector<T> _w;

    /// <summary>
    /// Initializes a new Time-Mixing layer.
    /// </summary>
    /// <param name="dModel">Model dimension (e.g., 512).
    /// <b>Default:</b> Typically matches embedding dimension of the model.</param>
    public TimeMixing(int dModel) : base(new[] { dModel }, new[] { dModel })
    {
        if (dModel <= 0)
            throw new ArgumentException("Model dimension must be positive", nameof(dModel));

        _dModel = dModel;

        // Initialize projection matrices
        _W_k = new DenseLayer<T>(dModel, dModel);
        _W_v = new DenseLayer<T>(dModel, dModel);
        _W_r = new DenseLayer<T>(dModel, dModel);
        _W_o = new DenseLayer<T>(dModel, dModel);

        // Initialize learnable parameters
        InitializeParameters();
    }

    private void InitializeParameters()
    {
        var random = new Random(42);

        // Interpolation parameters: initialized near 0.5 (equal mix)
        _mu_k = new Vector<T>(_dModel);
        _mu_v = new Vector<T>(_dModel);
        _mu_r = new Vector<T>(_dModel);

        for (int i = 0; i < _dModel; i++)
        {
            _mu_k[i] = NumOps.FromDouble(0.5 + (random.NextDouble() - 0.5) * 0.1);
            _mu_v[i] = NumOps.FromDouble(0.5 + (random.NextDouble() - 0.5) * 0.1);
            _mu_r[i] = NumOps.FromDouble(0.5 + (random.NextDouble() - 0.5) * 0.1);
        }

        // Time decay: initialized to small negative values
        // (exp of negative = decay factor < 1)
        _w = new Vector<T>(_dModel);
        for (int i = 0; i < _dModel; i++)
        {
            // Decay rate varies by dimension (different time scales)
            _w[i] = NumOps.FromDouble(-5.0 - random.NextDouble() * 2.0);
        }
    }

    public override bool SupportsTraining => true;

    /// <summary>
    /// Forward pass in parallel mode (for training).
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        return ForwardParallel(input);
    }

    /// <summary>
    /// Forward pass processing entire sequence in parallel.
    /// </summary>
    public Tensor<T> ForwardParallel(Tensor<T> x)
    {
        // x: (batch, seq_len, d_model)
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int dModel = x.Shape[2];

        if (dModel != _dModel)
            throw new ArgumentException($"Input dimension {dModel} doesn't match layer dimension {_dModel}");

        Tensor<T> output = new Tensor<T>(new[] { batch, seqLen, dModel });

        for (int b = 0; b < batch; b++)
        {
            // Shift inputs to get x_{t-1} (prepend zeros)
            Tensor<T> x_prev = new Tensor<T>(new[] { seqLen, dModel });
            for (int t = 1; t < seqLen; t++)
                for (int d = 0; d < dModel; d++)
                    x_prev[t, d] = x[b, t - 1, d];

            // Interpolate
            Tensor<T> x_k = InterpolateTensors(x_prev, x, b, _mu_k);
            Tensor<T> x_v = InterpolateTensors(x_prev, x, b, _mu_v);
            Tensor<T> x_r = InterpolateTensors(x_prev, x, b, _mu_r);

            // Project to keys, values, receptance
            Tensor<T> k = _W_k.Forward(x_k);
            Tensor<T> v = _W_v.Forward(x_v);
            Tensor<T> r = _W_r.Forward(x_r);

            // Apply sigmoid to receptance
            r = r.Transform((val, _) => Sigmoid(val));

            // WKV operation (parallel)
            Tensor<T> wkv = ComputeWKV_Parallel(k, v, seqLen);

            // Gate with receptance and project output
            Tensor<T> gated = ElementwiseMultiply(r, wkv);
            Tensor<T> out_b = _W_o.Forward(gated);

            // Copy to output
            for (int t = 0; t < seqLen; t++)
                for (int d = 0; d < dModel; d++)
                    output[b, t, d] = out_b[t, d];
        }

        return output;
    }

    /// <summary>
    /// Forward pass processing one token with state (for inference).
    /// </summary>
    public (Tensor<T> output, WKVState<T> newState) ForwardSequential(
        Tensor<T> x_t,
        WKVState<T> state)
    {
        // x_t: (batch, 1, d_model) - single token
        if (x_t.Shape[1] != 1)
            throw new ArgumentException("Sequential mode requires input with seq_len=1");

        int batch = x_t.Shape[0];
        int dModel = x_t.Shape[2];

        Tensor<T> output = new Tensor<T>(new[] { batch, 1, dModel });

        for (int b = 0; b < batch; b++)
        {
            Vector<T> x_curr = GetVector(x_t, b, 0);
            Vector<T> x_prev = state.PreviousInput;

            // Interpolate
            Vector<T> x_k = Interpolate(x_prev, x_curr, _mu_k);
            Vector<T> x_v = Interpolate(x_prev, x_curr, _mu_v);
            Vector<T> x_r = Interpolate(x_prev, x_curr, _mu_r);

            // Project
            Vector<T> k = _W_k.Forward(x_k.ToTensor()).ToVector();
            Vector<T> v = _W_v.Forward(x_v.ToTensor()).ToVector();
            Vector<T> r = _W_r.Forward(x_r.ToTensor()).ToVector();

            // Apply sigmoid to receptance
            r = r.Transform(val => Sigmoid(val));

            // WKV operation (sequential)
            Vector<T> wkv = ComputeWKV_Sequential(k, v, state);

            // Update state
            UpdateWKVState(state, k, v, x_curr);

            // Gate and project
            Vector<T> gated = ElementwiseMultiply(r, wkv);
            Vector<T> out_b = _W_o.Forward(gated.ToTensor()).ToVector();

            // Copy to output
            for (int d = 0; d < dModel; d++)
                output[b, 0, d] = out_b[d];
        }

        return (output, state);
    }

    /// <summary>
    /// Computes WKV in parallel mode.
    /// </summary>
    private Tensor<T> ComputeWKV_Parallel(Tensor<T> k, Tensor<T> v, int seqLen)
    {
        // k, v: (seq_len, d_model)
        Tensor<T> wkv = new Tensor<T>(new[] { seqLen, _dModel });

        for (int t = 0; t < seqLen; t++)
        {
            for (int d = 0; d < _dModel; d++)
            {
                T numerator = NumOps.Zero;
                T denominator = NumOps.Zero;

                for (int i = 0; i <= t; i++)
                {
                    // Time decay: exp(w × (t - i))
                    T decay_power = NumOps.Multiply(
                        _w[d],
                        NumOps.FromDouble(t - i));
                    T decay = NumOps.Exp(decay_power);

                    // Weight: decay × exp(k[i])
                    T weight = NumOps.Multiply(decay, NumOps.Exp(k[i, d]));

                    numerator = NumOps.Add(numerator,
                        NumOps.Multiply(weight, v[i, d]));
                    denominator = NumOps.Add(denominator, weight);
                }

                // Avoid division by zero
                if (NumOps.LessThan(NumOps.Abs(denominator), NumOps.FromDouble(1e-10)))
                    denominator = NumOps.FromDouble(1e-10);

                wkv[t, d] = NumOps.Divide(numerator, denominator);
            }
        }

        return wkv;
    }

    /// <summary>
    /// Computes WKV in sequential mode (updates running state).
    /// </summary>
    private Vector<T> ComputeWKV_Sequential(
        Vector<T> k_t,
        Vector<T> v_t,
        WKVState<T> state)
    {
        Vector<T> output = new Vector<T>(_dModel);

        for (int d = 0; d < _dModel; d++)
        {
            // Decay previous accumulator
            T decay = NumOps.Exp(_w[d]);
            state.Numerator[d] = NumOps.Multiply(state.Numerator[d], decay);
            state.Denominator[d] = NumOps.Multiply(state.Denominator[d], decay);

            // Add current contribution
            T weight = NumOps.Exp(k_t[d]);
            state.Numerator[d] = NumOps.Add(state.Numerator[d],
                NumOps.Multiply(weight, v_t[d]));
            state.Denominator[d] = NumOps.Add(state.Denominator[d], weight);

            // Compute output
            T denom = state.Denominator[d];
            if (NumOps.LessThan(NumOps.Abs(denom), NumOps.FromDouble(1e-10)))
                denom = NumOps.FromDouble(1e-10);

            output[d] = NumOps.Divide(state.Numerator[d], denom);
        }

        return output;
    }

    private void UpdateWKVState(WKVState<T> state, Vector<T> k, Vector<T> v, Vector<T> x_curr)
    {
        // Already updated in ComputeWKV_Sequential
        // Just update previous input
        state.PreviousInput = x_curr;
    }

    public WKVState<T> InitializeState()
    {
        return new WKVState<T>(_dModel);
    }

    private T Sigmoid(T x)
    {
        T neg_x = NumOps.Negate(x);
        T exp_neg = NumOps.Exp(neg_x);
        return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, exp_neg));
    }

    private Vector<T> Interpolate(Vector<T> prev, Vector<T> curr, Vector<T> mu)
    {
        Vector<T> result = new Vector<T>(prev.Length);
        for (int i = 0; i < prev.Length; i++)
        {
            T term1 = NumOps.Multiply(mu[i], prev[i]);
            T one_minus_mu = NumOps.Subtract(NumOps.One, mu[i]);
            T term2 = NumOps.Multiply(one_minus_mu, curr[i]);
            result[i] = NumOps.Add(term1, term2);
        }
        return result;
    }

    // Other required overrides...
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        throw new NotImplementedException("Backward pass for TimeMixing coming in Phase 2");
    }

    public override void UpdateParameters(T learningRate)
    {
        _W_k.UpdateParameters(learningRate);
        _W_v.UpdateParameters(learningRate);
        _W_r.UpdateParameters(learningRate);
        _W_o.UpdateParameters(learningRate);
    }

    public override Vector<T> GetParameters()
    {
        // Concatenate all parameters
        var allParams = new List<Vector<T>>
        {
            _W_k.GetParameters(),
            _W_v.GetParameters(),
            _W_r.GetParameters(),
            _W_o.GetParameters(),
            _mu_k,
            _mu_v,
            _mu_r,
            _w
        };

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
        _W_k.ResetState();
        _W_v.ResetState();
        _W_r.ResetState();
        _W_o.ResetState();
    }
}
```

### Phase 2: Channel-Mixing and RWKV Block Assembly

The Channel-Mixing implementation and full RWKV block assembly follow the same patterns shown above.

---

## Testing Strategy

### Critical Test: Parallel-Sequential Equivalence

```csharp
[Fact]
public void TimeMixing_ParallelSequentialEquivalence_ExactMatch()
{
    // Arrange
    int dModel = 32;
    int seqLen = 5;
    int batch = 1;

    var layer = new TimeMixing<double>(dModel);
    var input = CreateRandomTensor(batch, seqLen, dModel, seed: 42);

    // Act 1: Parallel mode
    var outputParallel = layer.ForwardParallel(input);

    // Act 2: Sequential mode
    var outputSequential = new Tensor<double>(new[] { batch, seqLen, dModel });
    var state = layer.InitializeState();

    for (int t = 0; t < seqLen; t++)
    {
        var input_t = input.Slice(1, t, t + 1);
        var (output_t, newState) = layer.ForwardSequential(input_t, state);

        for (int d = 0; d < dModel; d++)
            outputSequential[0, t, d] = output_t[0, 0, d];

        state = newState;
    }

    // Assert: Numerical equivalence (tolerance for floating-point errors)
    for (int t = 0; t < seqLen; t++)
    {
        for (int d = 0; d < dModel; d++)
        {
            double parallel = outputParallel[0, t, d];
            double sequential = outputSequential[0, t, d];
            double diff = Math.Abs(parallel - sequential);
            double relativeError = diff / (Math.Abs(parallel) + 1e-10);

            Assert.True(relativeError < 1e-4,
                $"Mismatch at position ({t}, {d}): " +
                $"parallel={parallel}, sequential={sequential}, " +
                $"diff={diff}, relativeError={relativeError}");
        }
    }
}
```

### Performance Test: Sequential Inference Efficiency

```csharp
[Fact]
public void RWKV_SequentialInference_ConstantMemory()
{
    // Arrange
    int dModel = 512;
    var model = new RWKVModel<double>(
        vocabSize: 50000,
        dModel: dModel,
        numLayers: 12);

    var state = model.InitializeState();
    long initialMemory = GC.GetTotalMemory(forceFullCollection: true);

    // Act: Generate 1000 tokens
    for (int i = 0; i < 1000; i++)
    {
        var input = new Tensor<double>(new[] { 1, 1, dModel });
        // Fill with random token embedding...

        var (output, newState) = model.ForwardSequential(input, state);
        state = newState;

        // Check memory every 100 tokens
        if (i % 100 == 0)
        {
            long currentMemory = GC.GetTotalMemory(forceFullCollection: false);
            long memoryGrowth = currentMemory - initialMemory;

            // Memory should be roughly constant (< 10% growth)
            Assert.True(memoryGrowth < initialMemory * 0.1,
                $"Memory grew by {memoryGrowth} bytes after {i} tokens");
        }
    }
}
```

---

## Common Pitfalls

### 1. Numerical Instability in WKV

**Problem:** Exponentials can overflow or underflow.

```csharp
// BAD: Can overflow for large k
T weight = NumOps.Exp(k[i, d]);

// GOOD: Use log-sum-exp trick
T max_k = FindMax(k); // Find maximum to prevent overflow
T weight = NumOps.Exp(NumOps.Subtract(k[i, d], max_k));
```

### 2. Forgetting to Reset State Between Sequences

**Problem:** State from one sequence contaminates the next during inference.

```csharp
// WRONG
for (var sequence in sequences)
{
    var (output, state) = model.ForwardSequential(sequence, state);
    // State carries over to next sequence!
}

// CORRECT
for (var sequence in sequences)
{
    var state = model.InitializeState(); // Fresh state for each sequence
    var (output, newState) = model.ForwardSequential(sequence, state);
}
```

### 3. Parameter Constraints Not Enforced

**Problem:** μ should be in [0, 1], w should be negative for stability.

```csharp
// Apply constraints during parameter updates
private void ConstrainParameters()
{
    // Clamp μ to [0, 1]
    for (int i = 0; i < _dModel; i++)
    {
        if (NumOps.LessThan(_mu_k[i], NumOps.Zero))
            _mu_k[i] = NumOps.Zero;
        if (NumOps.GreaterThan(_mu_k[i], NumOps.One))
            _mu_k[i] = NumOps.One;
        // Same for _mu_v, _mu_r

        // Ensure w is negative (for exponential decay)
        if (NumOps.GreaterThan(_w[i], NumOps.Zero))
            _w[i] = NumOps.Negate(_w[i]);
    }
}
```

### 4. Shape Mismatches in Sequential Mode

**Problem:** Sequential forward expects (batch, 1, d_model) not (batch, d_model).

```csharp
// WRONG
Vector<T> token = GetTokenEmbedding(tokenId);
var (output, state) = model.ForwardSequential(token, state); // Shape error!

// CORRECT
Vector<T> token = GetTokenEmbedding(tokenId);
Tensor<T> token_tensor = token.ToTensor().Reshape(1, 1, dModel);
var (output, state) = model.ForwardSequential(token_tensor, state);
```

---

## Summary

**What you've learned:**

1. **RWKV** combines RNN efficiency with Transformer training speed through dual-mode computation
2. **Time-Mixing** replaces attention with a recurrent weighted key-value mechanism
3. **Channel-Mixing** is a gated feed-forward network with temporal interpolation
4. **Dual implementation** allows the same model to run in parallel (training) or sequential (inference) mode
5. **Critical testing** ensures parallel and sequential modes produce identical results

**Key advantages:**
- O(N) memory instead of O(N²) for Transformers
- Constant inference memory regardless of context length
- Parallelizable training like Transformers
- Competitive performance on language tasks

**Next steps:**
- Implement full RWKV model with multiple layers
- Add backward pass for training
- Benchmark against Transformer baselines
- Optimize WKV computation for GPU

**Resources:**
- RWKV paper: https://arxiv.org/abs/2305.13048
- Official implementation: https://github.com/BlinkDL/RWKV-LM
- Blog post explanation: https://johanwind.github.io/2023/03/23/rwkv_overview.html

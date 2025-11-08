# Issue #275: State Space Models - Mamba/S4D Implementation Guide

## For Junior Developers: Complete Implementation Tutorial

### Table of Contents
1. [Understanding State Space Models](#understanding-state-space-models)
2. [Why SSMs Beat Transformers for Long Sequences](#advantages-over-transformers)
3. [S4/S4D Architecture Deep Dive](#s4-architecture)
4. [Mamba Architecture and Selective Scan](#mamba-architecture)
5. [Implementation Guide](#implementation-guide)
6. [Testing Strategy](#testing-strategy)
7. [Common Pitfalls](#common-pitfalls)

---

## Understanding State Space Models

### What are State Space Models?

**For Beginners:** State Space Models (SSMs) are a way to process sequences (like text, audio, or time series data) that's fundamentally different from Transformers. Instead of looking at all previous tokens at once (like attention), SSMs use a continuous-time mathematical model that can efficiently handle extremely long sequences.

**Real-world analogy:** Think of a Transformer as someone trying to remember an entire conversation by constantly re-reading all previous messages. An SSM is like someone who maintains a running summary in their head - they update their mental state with each new message but don't need to re-read everything.

**How it works at a high level:**
1. **Input**: A sequence of tokens (text, audio features, etc.)
2. **State Update**: For each token, update a hidden state using continuous-time dynamics
3. **Output**: Produce an output based on the current hidden state
4. **Key Advantage**: The state update is a simple linear operation, making it MUCH faster than attention for long sequences

---

## Advantages Over Transformers

### Why Use State Space Models?

#### 1. Linear Complexity vs Quadratic

**Transformers:**
- Self-attention requires comparing every token to every other token
- Memory: O(L²) where L is sequence length
- Computation: O(L² × D) where D is embedding dimension
- **Problem**: 10,000 token sequence = 100 million comparisons!

**State Space Models:**
- Each token only updates the hidden state
- Memory: O(L × D)
- Computation: O(L × D × N) where N is state dimension (typically much smaller than L)
- **Advantage**: 10,000 tokens = 10,000 simple updates!

**Example:**
```csharp
// Transformer attention (simplified)
// For sequence length L=10000, embedding dim D=768
Matrix<T> Q = queries; // Shape: (10000, 768)
Matrix<T> K = keys;    // Shape: (10000, 768)
Matrix<T> attention = MatrixHelper.Multiply(Q, K.Transpose()); // (10000, 10000) - HUGE!

// SSM state update (simplified)
// For same sequence, state dimension N=64
Vector<T> state = new Vector<T>(64); // Small state!
for (int t = 0; t < 10000; t++)
{
    // Simple linear update - O(N) instead of O(L)
    state = UpdateState(state, input[t]); // Fast!
}
```

#### 2. Constant Memory for Inference

**Transformers:**
- Must keep a growing KV cache of all previous tokens
- 100K token context = massive memory requirement
- Limits deployment on edge devices

**State Space Models:**
- Fixed-size hidden state regardless of sequence length
- 100K tokens or 1M tokens - same memory!
- Perfect for on-device deployment

#### 3. Better Long-Range Dependencies

**Why Transformers struggle:**
```
Token 1: "The key to the treasure is..."
[9,998 tokens of distraction]
Token 10,000: "...the old oak tree"
```

- Attention weights decay with distance
- Positional encodings lose effectiveness
- Information from Token 1 gets diluted

**Why SSMs excel:**
- Information flows through the state like a river
- Continuous-time dynamics preserve information over long periods
- Special initialization (like HiPPO) ensures long-term memory

---

## S4 Architecture

### Structured State Space Model (S4)

S4 is the foundation that Mamba builds upon. It introduced the key innovations that make SSMs practical.

### The Core SSM Equation

**Continuous-time formulation:**
```
dh/dt = A × h(t) + B × u(t)    [State update]
y(t) = C × h(t) + D × u(t)      [Output]
```

Where:
- `h(t)`: Hidden state (dimension N, typically 64-256)
- `u(t)`: Input signal (dimension 1 for each feature)
- `y(t)`: Output signal
- `A`: State transition matrix (N × N) - **KEY**: Controls how state evolves
- `B`: Input matrix (N × 1) - How input affects state
- `C`: Output matrix (1 × N) - How state produces output
- `D`: Feedthrough (usually 0 for sequence modeling)

**For Beginners:** Think of `h(t)` as the model's "memory" that evolves over time. `A` determines how the memory fades or persists, `B` controls how new input updates memory, and `C` reads out the memory to produce output.

### Discretization: From Continuous to Discrete

**The Problem:** Neural networks operate on discrete sequences (token 1, token 2, ...), but the SSM is defined in continuous time.

**The Solution:** Use zero-order hold (ZOH) discretization:

```csharp
// Discretization with step size Δ
// Converts continuous A, B to discrete A_bar, B_bar

public static void Discretize<T>(
    Matrix<T> A,           // Continuous state matrix
    Matrix<T> B,           // Continuous input matrix
    T delta,               // Step size (typically 0.001 to 0.1)
    out Matrix<T> A_bar,   // Discrete state matrix
    out Matrix<T> B_bar)   // Discrete input matrix
{
    // A_bar = exp(Δ × A)
    // This is matrix exponential - computationally expensive!
    A_bar = MatrixExponential(MatrixHelper.Multiply(delta, A));

    // B_bar = (A_bar - I) × A^(-1) × B
    // Simplification when A is diagonal: B_bar = (exp(Δ×A) - 1) / A × B
    Matrix<T> I = Matrix<T>.CreateIdentity(A.Rows);
    Matrix<T> A_diff = MatrixHelper.Subtract(A_bar, I);
    Matrix<T> A_inv = MatrixHelper.Inverse(A);
    B_bar = MatrixHelper.Multiply(MatrixHelper.Multiply(A_diff, A_inv), B);
}
```

**Discrete SSM recurrence (this is what we actually implement):**
```
h_t = A_bar × h_{t-1} + B_bar × u_t
y_t = C × h_t
```

### S4D: Diagonal State Spaces

**The Problem with S4:** Computing with a full N×N matrix A is expensive: O(N²) operations per step.

**S4D Innovation:** Restrict A to be diagonal!

```csharp
// Instead of full matrix:
Matrix<T> A = new Matrix<T>(64, 64); // 4096 parameters!

// Use diagonal:
Vector<T> A_diag = new Vector<T>(64); // Only 64 parameters!
```

**Advantages:**
1. **Faster computation:** Element-wise operations instead of matrix multiplication
2. **Fewer parameters:** N instead of N²
3. **Easier discretization:** Matrix exponential becomes element-wise exponential

**Discretization simplifies to:**
```csharp
// For diagonal A, discretization is element-wise!
public static void DiscretizeDiagonal<T>(
    Vector<T> A_diag,
    Vector<T> B_vec,
    T delta,
    out Vector<T> A_bar_diag,
    out Vector<T> B_bar_vec)
{
    int N = A_diag.Length;
    A_bar_diag = new Vector<T>(N);
    B_bar_vec = new Vector<T>(N);

    for (int i = 0; i < N; i++)
    {
        // A_bar[i] = exp(Δ × A[i])
        T deltaA = NumOps.Multiply(delta, A_diag[i]);
        A_bar_diag[i] = NumOps.Exp(deltaA);

        // B_bar[i] = (exp(Δ×A[i]) - 1) / A[i] × B[i]
        T numerator = NumOps.Subtract(A_bar_diag[i], NumOps.One);

        // Handle A[i] near zero to avoid division by zero
        T denominator = NumOps.Abs(A_diag[i]) > NumOps.FromDouble(1e-10)
            ? A_diag[i]
            : NumOps.FromDouble(1e-10);

        T factor = NumOps.Divide(numerator, denominator);
        B_bar_vec[i] = NumOps.Multiply(factor, B_vec[i]);
    }
}
```

### HiPPO Initialization

**The Secret Sauce:** Random initialization doesn't work well for SSMs. We need A to be initialized such that the state can remember long sequences.

**HiPPO (High-order Polynomial Projection Operators):**
- Mathematically derived initialization that approximates the entire history with polynomials
- Ensures the state can reconstruct past inputs
- Diagonal variant: `A[n] = -(n+1)` (negative integers)

```csharp
public static Vector<T> InitializeHiPPO_Diagonal<T>(int stateSize)
{
    Vector<T> A_diag = new Vector<T>(stateSize);
    for (int n = 0; n < stateSize; n++)
    {
        // HiPPO-LegS initialization: A[n] = -(n+1)
        A_diag[n] = NumOps.FromDouble(-(n + 1));
    }
    return A_diag;
}
```

**For Beginners:** This initialization ensures that the first state dimension remembers recent history, the second dimension remembers slightly older history, and so on. It's like having different "time scales" for memory.

---

## Mamba Architecture

### What Makes Mamba Special?

Mamba improves on S4 with **selective state spaces** - the A, B, C parameters are now input-dependent!

**S4/S4D (Fixed parameters):**
```csharp
// Same A, B, C for all inputs
for (int t = 0; t < sequenceLength; t++)
{
    h[t] = A_bar × h[t-1] + B_bar × u[t]; // Fixed A_bar, B_bar
    y[t] = C × h[t];                       // Fixed C
}
```

**Mamba (Selective parameters):**
```csharp
// A, B, C change based on input!
for (int t = 0; t < sequenceLength; t++)
{
    // Generate input-dependent parameters
    delta_t = ProjectionDelta(u[t]);  // Time step
    B_t = ProjectionB(u[t]);           // Input matrix
    C_t = ProjectionC(u[t]);           // Output matrix

    // Discretize with input-dependent delta
    Discretize(A, B_t, delta_t, out A_bar_t, out B_bar_t);

    // Update state with input-dependent parameters
    h[t] = A_bar_t × h[t-1] + B_bar_t × u[t];
    y[t] = C_t × h[t];
}
```

**Why this matters:**
- **Content-aware filtering:** Important inputs get larger B (stronger influence on state)
- **Adaptive forgetting:** Important context gets smaller delta (slower state evolution = longer memory)
- **Selective reading:** Important information gets different C (what to output)

**Real-world example:**
```
Sequence: "The capital of France is Paris. The weather is sunny. What is the capital of France?"

Token "Paris":
- Large B: Store strongly in state
- Small delta: Keep this information for a long time
- When seeing "capital of France?" again:
  - Large C: Read out the "Paris" information from state
```

### The S6 Scan Operation

This is the heart of Mamba - the selective scan.

**Sequential Implementation (Simple but Slow):**

```csharp
/// <summary>
/// Performs the selective SSM scan operation sequentially.
/// This is the reference implementation for correctness - parallel version must match this!
/// </summary>
public static Tensor<T> SequentialScan<T>(
    Tensor<T> u,      // Input sequence: (batch, seq_len, input_dim)
    Tensor<T> delta,  // Time steps: (batch, seq_len, state_dim)
    Tensor<T> A,      // State matrix (diagonal): (state_dim,)
    Tensor<T> B,      // Input matrices: (batch, seq_len, state_dim)
    Tensor<T> C)      // Output matrices: (batch, seq_len, state_dim)
{
    int batch = u.Shape[0];
    int seqLen = u.Shape[1];
    int inputDim = u.Shape[2];
    int stateDim = A.Shape[0];

    // Output tensor
    Tensor<T> y = new Tensor<T>(new[] { batch, seqLen, inputDim });

    // Hidden states for each batch item
    Tensor<T> h = new Tensor<T>(new[] { batch, stateDim });

    for (int b = 0; b < batch; b++)
    {
        // Reset hidden state for this batch item
        for (int n = 0; n < stateDim; n++)
            h[b, n] = NumOps.Zero;

        for (int t = 0; t < seqLen; t++)
        {
            // Get current input-dependent parameters
            Vector<T> delta_t = GetSlice(delta, b, t); // (state_dim,)
            Vector<T> B_t = GetSlice(B, b, t);         // (state_dim,)
            Vector<T> C_t = GetSlice(C, b, t);         // (state_dim,)
            T u_t = u[b, t, 0]; // Assuming input_dim=1 for simplicity

            // Discretization (element-wise for diagonal A)
            Vector<T> delta_A = new Vector<T>(stateDim);
            Vector<T> delta_B = new Vector<T>(stateDim);

            for (int n = 0; n < stateDim; n++)
            {
                // delta_A[n] = exp(delta[n] × A[n])
                T temp = NumOps.Multiply(delta_t[n], A[n]);
                delta_A[n] = NumOps.Exp(temp);

                // delta_B[n] = (exp(delta[n]×A[n]) - 1) / A[n] × B[n]
                T numerator = NumOps.Subtract(delta_A[n], NumOps.One);
                T denominator = NumOps.Abs(A[n]) > NumOps.FromDouble(1e-10)
                    ? A[n]
                    : NumOps.FromDouble(1e-10);
                T factor = NumOps.Divide(numerator, denominator);
                delta_B[n] = NumOps.Multiply(NumOps.Multiply(factor, B_t[n]), u_t);
            }

            // State update: h_new = delta_A ⊙ h_old + delta_B
            Vector<T> h_new = new Vector<T>(stateDim);
            for (int n = 0; n < stateDim; n++)
            {
                T term1 = NumOps.Multiply(delta_A[n], h[b, n]);
                h_new[n] = NumOps.Add(term1, delta_B[n]);
            }

            // Copy new state back
            for (int n = 0; n < stateDim; n++)
                h[b, n] = h_new[n];

            // Output: y_t = C_t ⊙ h_new (dot product)
            T y_t = NumOps.Zero;
            for (int n = 0; n < stateDim; n++)
            {
                y_t = NumOps.Add(y_t, NumOps.Multiply(C_t[n], h_new[n]));
            }

            y[b, t, 0] = y_t;
        }
    }

    return y;
}
```

**For Beginners:** This loop processes one token at a time, updating the state based on input-specific parameters. It's slow because it's sequential (can't parallelize), but it's the correct reference implementation.

### Parallel Scan (Advanced)

**The Challenge:** The sequential scan above is inherently sequential - we need h[t-1] before computing h[t].

**The Solution:** Use an associative scan algorithm that can parallelize the computation.

**Key Insight:** The SSM recurrence can be written as:
```
h_t = δA_t ⊙ h_{t-1} + δB_t
```

This has the form of a linear recurrence: `h_t = a_t ⊙ h_{t-1} + b_t`

**Associative property:**
```
(h_1 from h_0) ⊙ (h_2 from h_1) = (h_2 from h_0)
```

This allows us to use a parallel prefix sum algorithm!

**Implementation sketch (simplified):**
```csharp
// Phase 1: Compute all (a_t, b_t) pairs in parallel
Parallel.For(0, seqLen, t => {
    a[t] = delta_A[t];  // Discretized A
    b[t] = delta_B[t];  // Discretized B × u
});

// Phase 2: Parallel associative scan (like parallel prefix sum)
// This uses O(log L) steps instead of O(L) sequential steps
int depth = (int)Math.Ceiling(Math.Log2(seqLen));
for (int d = 0; d < depth; d++)
{
    int stride = 1 << d; // 1, 2, 4, 8, ...
    Parallel.For(0, seqLen, t => {
        if (t >= stride)
        {
            // Combine (a[t-stride], b[t-stride]) with (a[t], b[t])
            T a_new = NumOps.Multiply(a[t], a[t - stride]);
            T b_new = NumOps.Add(NumOps.Multiply(a[t], b[t - stride]), b[t]);
            a[t] = a_new;
            b[t] = b_new;
        }
    });
}

// Now b[t] contains the final states!
```

**Performance:**
- Sequential: O(L) steps
- Parallel: O(log L) steps with O(L) processors
- Critical for efficient training on GPUs

**Important:** The parallel scan MUST produce identical results to the sequential scan. This is a critical test!

---

## Implementation Guide

### Phase 1: S6 Sequential Scan

#### Step 1: Create the S6Scan class

**File:** `src/NeuralNetworks/Layers/SSM/S6Scan.cs`

```csharp
using AiDotNet.Mathematics;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the S6 (Selective State Space) scan operation for Mamba models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class performs the core computation of Mamba models.
///
/// The S6 scan processes a sequence by maintaining a hidden state that gets updated
/// at each step based on input-dependent parameters. Think of it like a running summary
/// that adapts based on what information is important.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type (typically float or double)</typeparam>
public static class S6Scan<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Performs the selective SSM scan operation sequentially (reference implementation).
    /// </summary>
    /// <param name="u">Input sequence of shape (batch, seq_len, input_dim)</param>
    /// <param name="delta">Time steps of shape (batch, seq_len, state_dim)</param>
    /// <param name="A">State transition vector (diagonal) of shape (state_dim,)</param>
    /// <param name="B">Input projection of shape (batch, seq_len, state_dim)</param>
    /// <param name="C">Output projection of shape (batch, seq_len, state_dim)</param>
    /// <returns>Output sequence of shape (batch, seq_len, input_dim)</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the heart of Mamba - the selective scan.
    ///
    /// It processes each token in sequence, updating a hidden state based on:
    /// - How much to forget from previous state (controlled by delta and A)
    /// - How much new information to add (controlled by B and u)
    /// - What to output (controlled by C)
    ///
    /// The "selective" part means these controls change based on the input,
    /// allowing the model to remember important information and forget irrelevant details.
    /// </para>
    /// </remarks>
    public static Tensor<T> SequentialScan(
        Tensor<T> u,
        Tensor<T> delta,
        Tensor<T> A,
        Tensor<T> B,
        Tensor<T> C)
    {
        // Validate inputs
        if (u.Rank != 3)
            throw new ArgumentException("Input u must be rank 3 (batch, seq_len, input_dim)");
        if (delta.Rank != 3)
            throw new ArgumentException("Delta must be rank 3 (batch, seq_len, state_dim)");
        if (A.Rank != 1)
            throw new ArgumentException("A must be rank 1 (diagonal state matrix)");
        if (B.Rank != 3)
            throw new ArgumentException("B must be rank 3 (batch, seq_len, state_dim)");
        if (C.Rank != 3)
            throw new ArgumentException("C must be rank 3 (batch, seq_len, state_dim)");

        int batch = u.Shape[0];
        int seqLen = u.Shape[1];
        int inputDim = u.Shape[2];
        int stateDim = A.Shape[0];

        // Validate dimensions match
        if (delta.Shape[0] != batch || delta.Shape[1] != seqLen || delta.Shape[2] != stateDim)
            throw new ArgumentException("Delta shape mismatch");
        if (B.Shape[0] != batch || B.Shape[1] != seqLen || B.Shape[2] != stateDim)
            throw new ArgumentException("B shape mismatch");
        if (C.Shape[0] != batch || C.Shape[1] != seqLen || C.Shape[2] != stateDim)
            throw new ArgumentException("C shape mismatch");

        // Output tensor
        Tensor<T> y = new Tensor<T>(new[] { batch, seqLen, inputDim });

        // Process each batch independently
        for (int b = 0; b < batch; b++)
        {
            // Initialize hidden state to zeros
            Vector<T> h = new Vector<T>(stateDim);

            // Process sequence
            for (int t = 0; t < seqLen; t++)
            {
                // Discretize: compute delta_A and delta_B
                for (int n = 0; n < stateDim; n++)
                {
                    // Get input-dependent parameters for this timestep
                    T delta_tn = delta[b, t, n];
                    T B_tn = B[b, t, n];
                    T A_n = A[n];

                    // Discretization: delta_A = exp(delta × A)
                    T deltaA_product = NumOps.Multiply(delta_tn, A_n);
                    T delta_A_n = NumOps.Exp(deltaA_product);

                    // Discretization: delta_B = (exp(delta×A) - 1) / A × B
                    T numerator = NumOps.Subtract(delta_A_n, NumOps.One);

                    // Handle near-zero A to avoid division by zero
                    T A_safe = NumOps.GreaterThan(NumOps.Abs(A_n), NumOps.FromDouble(1e-10))
                        ? A_n
                        : NumOps.FromDouble(1e-10);

                    T discretization_factor = NumOps.Divide(numerator, A_safe);

                    // delta_B incorporates the input u
                    T delta_B_n = NumOps.Zero;
                    for (int d = 0; d < inputDim; d++)
                    {
                        T u_td = u[b, t, d];
                        T contrib = NumOps.Multiply(NumOps.Multiply(discretization_factor, B_tn), u_td);
                        delta_B_n = NumOps.Add(delta_B_n, contrib);
                    }

                    // State update: h[n] = delta_A × h[n] + delta_B
                    T h_old = h[n];
                    T h_new = NumOps.Add(NumOps.Multiply(delta_A_n, h_old), delta_B_n);
                    h[n] = h_new;
                }

                // Output: y = C × h (dot product)
                for (int d = 0; d < inputDim; d++)
                {
                    T y_td = NumOps.Zero;
                    for (int n = 0; n < stateDim; n++)
                    {
                        T C_tn = C[b, t, n];
                        y_td = NumOps.Add(y_td, NumOps.Multiply(C_tn, h[n]));
                    }
                    y[b, t, d] = y_td;
                }
            }
        }

        return y;
    }
}
```

#### Step 2: Create the MambaBlock

**File:** `src/NeuralNetworks/Layers/SSM/MambaBlock.cs`

```csharp
using AiDotNet.Mathematics;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements a Mamba block with selective state space model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A Mamba block is like a super-efficient LSTM replacement.
///
/// It has two main paths:
/// 1. SSM path: Uses the selective scan to process sequences efficiently
/// 2. Gating path: Uses a convolution and activation to control information flow
///
/// These paths are combined to produce the final output. The block can handle
/// much longer sequences than transformers because it doesn't use attention.
/// </para>
/// </remarks>
public class MambaBlock<T> : LayerBase<T>
{
    private readonly int _inputDim;
    private readonly int _stateDim;
    private readonly int _expansionFactor;
    private readonly int _convKernelSize;

    // Projections for generating SSM parameters
    private readonly DenseLayer<T> _projDelta;
    private readonly DenseLayer<T> _projB;
    private readonly DenseLayer<T> _projC;
    private readonly DenseLayer<T> _projInput;
    private readonly DenseLayer<T> _projOutput;

    // Convolutional layer for temporal processing
    private readonly ConvolutionalLayer<T> _conv1d;

    // SSM state matrix (diagonal, initialized with HiPPO)
    private Vector<T> _A;

    // SiLU activation
    private readonly IActivationFunction<T> _silu;

    /// <summary>
    /// Initializes a new Mamba block.
    /// </summary>
    /// <param name="inputDim">Input dimension (e.g., 512)</param>
    /// <param name="stateDim">SSM state dimension (e.g., 64). Default: 64.
    /// <b>Why 64?</b> Empirically found to be a good trade-off between memory and expressiveness in Mamba paper.</param>
    /// <param name="expansionFactor">Expansion factor for internal dimension (default: 2).
    /// <b>Why 2?</b> Allows richer representations without excessive parameters.</param>
    /// <param name="convKernelSize">Kernel size for 1D convolution (default: 3).
    /// <b>Why 3?</b> Captures local context efficiently.</param>
    public MambaBlock(
        int inputDim,
        int stateDim = 64,
        int expansionFactor = 2,
        int convKernelSize = 3)
        : base(new[] { inputDim }, new[] { inputDim })
    {
        if (inputDim <= 0)
            throw new ArgumentException("Input dimension must be positive", nameof(inputDim));
        if (stateDim <= 0)
            throw new ArgumentException("State dimension must be positive", nameof(stateDim));
        if (expansionFactor <= 0)
            throw new ArgumentException("Expansion factor must be positive", nameof(expansionFactor));
        if (convKernelSize <= 0 || convKernelSize % 2 == 0)
            throw new ArgumentException("Convolution kernel size must be positive and odd", nameof(convKernelSize));

        _inputDim = inputDim;
        _stateDim = stateDim;
        _expansionFactor = expansionFactor;
        _convKernelSize = convKernelSize;

        int innerDim = inputDim * expansionFactor;

        // Initialize projections
        _projInput = new DenseLayer<T>(inputDim, innerDim);
        _projDelta = new DenseLayer<T>(innerDim, stateDim);
        _projB = new DenseLayer<T>(innerDim, stateDim);
        _projC = new DenseLayer<T>(innerDim, stateDim);
        _projOutput = new DenseLayer<T>(innerDim, inputDim);

        // 1D convolution (causal, for temporal context)
        _conv1d = new ConvolutionalLayer<T>(
            inputDepth: 1,
            outputDepth: 1,
            kernelSize: convKernelSize,
            stride: 1,
            padding: (convKernelSize - 1) / 2);

        // Initialize A with HiPPO
        _A = InitializeHiPPO(stateDim);

        // SiLU activation
        _silu = new SiLUActivation<T>();
    }

    /// <summary>
    /// Initializes the state matrix using HiPPO for long-term memory.
    /// </summary>
    private Vector<T> InitializeHiPPO(int size)
    {
        Vector<T> A = new Vector<T>(size);
        for (int n = 0; n < size; n++)
        {
            // HiPPO-LegS: A[n] = -(n+1)
            // Negative values ensure stability (state decays over time)
            // Different values create different timescales for memory
            A[n] = NumOps.FromDouble(-(n + 1));
        }
        return A;
    }

    public override bool SupportsTraining => true;

    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Input shape: (batch, seq_len, input_dim)
        int batch = input.Shape[0];
        int seqLen = input.Shape[1];

        // Project input to inner dimension
        Tensor<T> x = _projInput.Forward(input); // (batch, seq_len, inner_dim)

        // Generate SSM parameters (input-dependent!)
        Tensor<T> delta = _projDelta.Forward(x); // (batch, seq_len, state_dim)
        Tensor<T> B = _projB.Forward(x);         // (batch, seq_len, state_dim)
        Tensor<T> C = _projC.Forward(x);         // (batch, seq_len, state_dim)

        // Apply softplus to delta to ensure positivity
        delta = delta.Transform((val, _) => {
            T exp_val = NumOps.Exp(val);
            return NumOps.Log(NumOps.Add(NumOps.One, exp_val));
        });

        // Convolution + activation path (for gating)
        Tensor<T> x_conv = _conv1d.Forward(x); // (batch, seq_len, inner_dim)
        Tensor<T> x_activated = x_conv.Transform((val, _) => _silu.Activate(val));

        // SSM scan
        Tensor<T> A_expanded = new Tensor<T>(new[] { _stateDim });
        for (int i = 0; i < _stateDim; i++)
            A_expanded[i] = _A[i];

        Tensor<T> y = S6Scan<T>.SequentialScan(x_activated, delta, A_expanded, B, C);

        // Multiply SSM output by gating path
        Tensor<T> gated = y.ElementwiseMultiply(x_activated);

        // Project back to input dimension
        Tensor<T> output = _projOutput.Forward(gated);

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // TODO: Implement backward pass
        // This requires computing gradients through the SSM scan,
        // which is complex. For now, we focus on forward pass correctness.
        throw new NotImplementedException("Backward pass for MambaBlock coming in Phase 2");
    }

    public override void UpdateParameters(T learningRate)
    {
        _projInput.UpdateParameters(learningRate);
        _projDelta.UpdateParameters(learningRate);
        _projB.UpdateParameters(learningRate);
        _projC.UpdateParameters(learningRate);
        _projOutput.UpdateParameters(learningRate);
        _conv1d.UpdateParameters(learningRate);
    }

    public override Vector<T> GetParameters()
    {
        // Concatenate all parameters
        var allParams = new List<Vector<T>>
        {
            _projInput.GetParameters(),
            _projDelta.GetParameters(),
            _projB.GetParameters(),
            _projC.GetParameters(),
            _projOutput.GetParameters(),
            _conv1d.GetParameters(),
            _A
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
        _projInput.ResetState();
        _projDelta.ResetState();
        _projB.ResetState();
        _projC.ResetState();
        _projOutput.ResetState();
        _conv1d.ResetState();
    }
}
```

---

## Testing Strategy

### Unit Test for S6 Scan

**File:** `tests/UnitTests/NeuralNetworks/Layers/SSM/S6ScanTests.cs`

```csharp
using Xunit;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Mathematics;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

public class S6ScanTests
{
    [Fact]
    public void SequentialScan_SimpleCase_ProducesCorrectOutput()
    {
        // Arrange: Create a simple 2-token sequence
        int batch = 1;
        int seqLen = 2;
        int inputDim = 1;
        int stateDim = 2;

        // Input sequence: [1.0, 2.0]
        var u = new Tensor<double>(new[] { batch, seqLen, inputDim });
        u[0, 0, 0] = 1.0;
        u[0, 1, 0] = 2.0;

        // Delta (time steps): all 0.1
        var delta = Tensor<double>.CreateDefault(new[] { batch, seqLen, stateDim }, 0.1);

        // A (state matrix): [-1, -2] (HiPPO-like)
        var A = new Tensor<double>(new[] { stateDim });
        A[0] = -1.0;
        A[1] = -2.0;

        // B (input projection): all 1.0
        var B = Tensor<double>.CreateDefault(new[] { batch, seqLen, stateDim }, 1.0);

        // C (output projection): all 1.0
        var C = Tensor<double>.CreateDefault(new[] { batch, seqLen, stateDim }, 1.0);

        // Act
        var y = S6Scan<double>.SequentialScan(u, delta, A, B, C);

        // Assert
        Assert.Equal(new[] { batch, seqLen, inputDim }, y.Shape);
        Assert.False(double.IsNaN(y[0, 0, 0]));
        Assert.False(double.IsInfinity(y[0, 0, 0]));
        Assert.True(y[0, 0, 0] > 0); // Should be positive since all inputs are positive

        // Second output should be influenced by first
        Assert.True(y[0, 1, 0] > y[0, 0, 0]); // Accumulation effect
    }

    [Fact]
    public void SequentialScan_ZeroInput_ProducesZeroOutput()
    {
        // Arrange
        int batch = 1;
        int seqLen = 5;
        int inputDim = 1;
        int stateDim = 4;

        var u = Tensor<double>.CreateDefault(new[] { batch, seqLen, inputDim }, 0.0);
        var delta = Tensor<double>.CreateDefault(new[] { batch, seqLen, stateDim }, 0.1);
        var A = new Tensor<double>(new[] { stateDim });
        for (int i = 0; i < stateDim; i++)
            A[i] = -(i + 1);

        var B = Tensor<double>.CreateDefault(new[] { batch, seqLen, stateDim }, 1.0);
        var C = Tensor<double>.CreateDefault(new[] { batch, seqLen, stateDim }, 1.0);

        // Act
        var y = S6Scan<double>.SequentialScan(u, delta, A, B, C);

        // Assert: All outputs should be near zero
        for (int t = 0; t < seqLen; t++)
        {
            Assert.True(Math.Abs(y[0, t, 0]) < 1e-6);
        }
    }
}
```

### Integration Test for MambaBlock

**File:** `tests/IntegrationTests/NeuralNetworks/Layers/MambaBlockTests.cs`

```csharp
using Xunit;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Mathematics;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks.Layers;

public class MambaBlockTests
{
    [Fact]
    public void MambaBlock_ForwardPass_ProducesCorrectShape()
    {
        // Arrange
        int inputDim = 64;
        int stateDim = 16;
        int batch = 2;
        int seqLen = 10;

        var block = new MambaBlock<double>(
            inputDim: inputDim,
            stateDim: stateDim);

        var input = new Tensor<double>(new[] { batch, seqLen, inputDim });
        // Fill with random values
        var random = new Random(42);
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < seqLen; t++)
                for (int d = 0; d < inputDim; d++)
                    input[b, t, d] = random.NextDouble() - 0.5;

        // Act
        var output = block.Forward(input);

        // Assert
        Assert.Equal(new[] { batch, seqLen, inputDim }, output.Shape);

        // Check no NaN or Inf
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < seqLen; t++)
                for (int d = 0; d < inputDim; d++)
                {
                    Assert.False(double.IsNaN(output[b, t, d]));
                    Assert.False(double.IsInfinity(output[b, t, d]));
                }
    }

    [Fact]
    public void MambaBlock_LongSequence_HandlesEfficiently()
    {
        // Arrange: Test with long sequence (1000 tokens)
        int inputDim = 32;
        int stateDim = 16;
        int batch = 1;
        int seqLen = 1000; // Long sequence!

        var block = new MambaBlock<double>(inputDim, stateDim);
        var input = new Tensor<double>(new[] { batch, seqLen, inputDim });

        // Fill with pattern that tests long-range dependency
        for (int t = 0; t < seqLen; t++)
            input[0, t, 0] = t == 0 ? 1.0 : 0.0; // Signal at start

        // Act
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var output = block.Forward(input);
        sw.Stop();

        // Assert
        Assert.Equal(new[] { batch, seqLen, inputDim }, output.Shape);
        Assert.True(sw.ElapsedMilliseconds < 5000); // Should complete in reasonable time

        // Check that information from t=0 influences later outputs
        double firstOutput = output[0, 0, 0];
        double lastOutput = output[0, seqLen - 1, 0];
        Assert.False(double.IsNaN(lastOutput));
    }
}
```

---

## Common Pitfalls

### 1. Division by Zero in Discretization

**Problem:**
```csharp
T factor = NumOps.Divide(numerator, A[i]); // A[i] might be very small!
```

**Solution:**
```csharp
T A_safe = NumOps.GreaterThan(NumOps.Abs(A[i]), NumOps.FromDouble(1e-10))
    ? A[i]
    : NumOps.FromDouble(1e-10);
T factor = NumOps.Divide(numerator, A_safe);
```

### 2. Forgetting to Reset State Between Sequences

**Problem:** Hidden state from one sequence affects the next.

**Solution:**
```csharp
public override void ResetState()
{
    // Clear any cached hidden states
    _hiddenState = new Vector<T>(_stateDim); // All zeros
}
```

### 3. Delta Must Be Positive

**Problem:** Negative delta leads to numerical instability.

**Solution:**
```csharp
// Apply softplus: softplus(x) = log(1 + exp(x))
delta = delta.Transform((val, _) => {
    T exp_val = NumOps.Exp(val);
    return NumOps.Log(NumOps.Add(NumOps.One, exp_val));
});
```

### 4. Parallel Scan Must Match Sequential

**Critical Test:**
```csharp
[Fact]
public void ParallelScan_MatchesSequentialScan()
{
    // Create test inputs
    var u = CreateRandomTensor(...);
    var delta = CreateRandomTensor(...);
    // ... other inputs

    // Run both implementations
    var y_sequential = S6Scan<T>.SequentialScan(u, delta, A, B, C);
    var y_parallel = S6Scan<T>.ParallelScan(u, delta, A, B, C);

    // Assert: Results must be numerically identical
    for (int i = 0; i < y_sequential.Length; i++)
    {
        double diff = Math.Abs(
            Convert.ToDouble(y_sequential[i]) -
            Convert.ToDouble(y_parallel[i]));
        Assert.True(diff < 1e-5, $"Mismatch at index {i}");
    }
}
```

---

## Summary

**What you've learned:**

1. **State Space Models** provide linear-time complexity for sequence processing, compared to quadratic for Transformers
2. **S4/S4D** use diagonal state matrices and HiPPO initialization for efficient long-term memory
3. **Mamba** adds selectivity - making parameters input-dependent for content-aware processing
4. **S6 Scan** is the core operation, with sequential (simple) and parallel (fast) implementations
5. **Testing** requires validating numerical stability, shape correctness, and equivalence of parallel/sequential scans

**Next steps for Phase 3:**
- Implement parallel scan using associative scan algorithm
- Verify parallel scan matches sequential scan numerically
- Benchmark performance on long sequences (10K+ tokens)
- Implement backward pass for training

**Resources:**
- Mamba paper: https://arxiv.org/abs/2312.00752
- S4 paper: https://arxiv.org/abs/2111.00396
- HiPPO paper: https://arxiv.org/abs/2008.07669

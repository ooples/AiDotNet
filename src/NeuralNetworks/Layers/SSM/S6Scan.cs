using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Provides reusable S6 (Selective Structured State Space Sequence) scan operations for Mamba-family architectures.
/// </summary>
/// <remarks>
/// <para>
/// S6 is the selective scan algorithm at the heart of Mamba. It implements the core SSM recurrence:
/// <code>
///   h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
///   y_t = C_t * h_t + D * x_t
/// </code>
/// where A_bar and B_bar are discretized via the Zero-Order Hold (ZOH) method using input-dependent
/// delta (timestep) parameters. The "selective" aspect means that delta, B, and C are all functions
/// of the input, allowing the model to dynamically control information flow.
/// </para>
/// <para>
/// This static utility class extracts the scan operations from MambaBlock so they can be reused
/// by other SSM layers (Mamba2, hybrid architectures, etc.) without code duplication.
/// </para>
/// <para><b>For Beginners:</b> This class contains the math that makes Mamba work.
///
/// Imagine reading a book and keeping notes:
/// - At each word (timestep), you update your notes (hidden state h)
/// - How much you update depends on the current word (selective mechanism)
/// - Your output is a summary based on your current notes
///
/// The "scan" processes the entire sequence step by step, updating state at each position.
/// This class provides two ways to do this:
/// - Sequential scan: processes one step at a time (simple, always correct)
/// - Parallel scan: processes multiple steps simultaneously using a prefix-sum trick (faster on GPUs)
/// </para>
/// <para>
/// <b>Reference:</b> Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2024.
/// https://arxiv.org/abs/2312.00752
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public static class S6Scan<T>
{
    private static IEngine Engine => AiDotNetEngine.Current;
    private static INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Performs the forward pass of the S6 selective scan using sequential processing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The sequential scan processes each timestep in order, maintaining a running hidden state.
    /// Within each timestep, all computation uses Engine tensor operations for hardware acceleration.
    /// Only the time loop itself is sequential (due to the recurrence h_t depends on h_{t-1}).
    /// </para>
    /// <para><b>For Beginners:</b> This reads through the sequence one position at a time,
    /// updating a memory (hidden state) at each step and producing an output. It's like reading
    /// a sentence word by word and keeping track of what you've read so far.</para>
    /// </remarks>
    /// <param name="x">Input tensor [batch, seqLen, innerDim] (post-conv, post-SiLU).</param>
    /// <param name="delta">Timestep parameter [batch, seqLen, innerDim] (after softplus).</param>
    /// <param name="aLog">Log of the A parameter [innerDim, stateDim].</param>
    /// <param name="b">B parameter [batch, seqLen, stateDim].</param>
    /// <param name="c">C parameter [batch, seqLen, stateDim].</param>
    /// <param name="dParam">D skip connection parameter [innerDim].</param>
    /// <param name="batchSize">Number of sequences in the batch.</param>
    /// <param name="seqLen">Length of each sequence.</param>
    /// <param name="innerDimension">Inner dimension of the SSM.</param>
    /// <param name="stateDimension">State dimension of the SSM.</param>
    /// <param name="initialState">Optional initial hidden state [batch, innerDim, stateDim]. If null, starts from zeros.</param>
    /// <returns>A tuple of (output [batch, seqLen, innerDim], hiddenStates [batch, seqLen+1, innerDim, stateDim]).</returns>
    public static (Tensor<T> Output, Tensor<T> HiddenStates) SequentialScanForward(
        Tensor<T> x, Tensor<T> delta, Tensor<T> aLog, Tensor<T> b, Tensor<T> c, Tensor<T> dParam,
        int batchSize, int seqLen, int innerDimension, int stateDimension,
        Tensor<T>? initialState = null)
    {
        if (x.Rank != 3 || x.Shape[0] != batchSize || x.Shape[1] != seqLen || x.Shape[2] != innerDimension)
            throw new ArgumentException($"x must be [batch={batchSize}, seqLen={seqLen}, innerDim={innerDimension}], got shape [{string.Join(",", x.Shape.ToArray())}].");
        if (delta.Rank != 3 || delta.Shape[0] != batchSize || delta.Shape[1] != seqLen)
            throw new ArgumentException($"delta shape mismatch: expected [batch={batchSize}, seqLen={seqLen}, ...], got [{string.Join(",", delta.Shape.ToArray())}].");
        if (b.Rank != 3 || b.Shape[0] != batchSize || b.Shape[1] != seqLen || b.Shape[2] != stateDimension)
            throw new ArgumentException($"b must be [batch={batchSize}, seqLen={seqLen}, stateDim={stateDimension}], got [{string.Join(",", b.Shape.ToArray())}].");
        if (c.Rank != 3 || c.Shape[0] != batchSize || c.Shape[1] != seqLen || c.Shape[2] != stateDimension)
            throw new ArgumentException($"c must be [batch={batchSize}, seqLen={seqLen}, stateDim={stateDimension}], got [{string.Join(",", c.Shape.ToArray())}].");

        var output = new Tensor<T>(new[] { batchSize, seqLen, innerDimension });

        // Pre-compute A = -exp(A_log) as tensor: [innerDim, stateDim]
        var negA = Engine.TensorNegate(Engine.TensorExp(aLog));
        // Expand for broadcasting: [1, innerDim, stateDim]
        var negA3D = Engine.TensorExpandDims(negA, 0);

        // D parameter for skip connection: [1, innerDim] for broadcasting
        var D2D = dParam.Reshape(1, innerDimension);

        // Hidden state: [batch, innerDim, stateDim] - use initial state or zeros
        Tensor<T> h;
        if (initialState != null)
        {
            // Copy initial state so we don't mutate the caller's tensor
            h = new Tensor<T>(new[] { batchSize, innerDimension, stateDimension });
            h = Engine.TensorAdd(h, initialState);
        }
        else
        {
            h = new Tensor<T>(new[] { batchSize, innerDimension, stateDimension });
        }

        // Store all hidden states for backward pass: [batch, seqLen+1, innerDim, stateDim]
        var allHiddenStates = new Tensor<T>(new[] { batchSize, seqLen + 1, innerDimension, stateDimension });

        // Store initial state at index 0
        if (initialState != null)
        {
            allHiddenStates.SetSlice(1, 0, initialState);
        }

        // Time loop - only sequential dependency
        for (int t = 0; t < seqLen; t++)
        {
            // Extract slices for this timestep — clone to ensure contiguous memory
            var x_t = x.GetSliceAlongDimension(t, 1).Clone();         // [batch, innerDim]
            var delta_t = delta.GetSliceAlongDimension(t, 1).Clone();  // [batch, innerDim]
            var B_t = b.GetSliceAlongDimension(t, 1).Clone();          // [batch, stateDim]
            var C_t = c.GetSliceAlongDimension(t, 1).Clone();          // [batch, stateDim]

            // Expand dimensions for broadcasting to [batch, innerDim, stateDim]
            var delta_t_3D = Engine.TensorExpandDims(delta_t, 2);  // [batch, innerDim, 1]
            var B_t_3D = Engine.TensorExpandDims(B_t, 1);          // [batch, 1, stateDim]
            var C_t_3D = Engine.TensorExpandDims(C_t, 1);          // [batch, 1, stateDim]
            var x_t_3D = Engine.TensorExpandDims(x_t, 2);          // [batch, innerDim, 1]

            // Discretize: A_bar = exp(delta * A) where A = -exp(A_log)
            // Clone broadcast results to ensure contiguous memory layout
            var deltaA = Engine.TensorBroadcastMultiply(delta_t_3D, negA3D).Clone();
            var A_bar = Engine.TensorExp(deltaA).Clone();

            // B_bar * x = delta * B * x (Euler discretization for B)
            var deltaB = Engine.TensorBroadcastMultiply(delta_t_3D, B_t_3D).Clone();
            var Bbar_x = Engine.TensorBroadcastMultiply(deltaB, x_t_3D).Clone();

            // State update: h = A_bar * h + B_bar * x (all Engine tensor ops)
            h = Engine.TensorAdd(Engine.TensorMultiply(A_bar, h), Bbar_x);

            // Output: y_t = sum_n(C * h) + D * x (Engine reduction + broadcast)
            var Ch = Engine.TensorBroadcastMultiply(C_t_3D, h).Clone();
            var y_t = Engine.ReduceSum(Ch, new int[] { 2 });                   // [batch, innerDim]
            var Dx = Engine.TensorBroadcastMultiply(D2D, x_t);                // [batch, innerDim]
            y_t = Engine.TensorAdd(y_t, Dx);

            // Store hidden state and output
            allHiddenStates.SetSlice(1, t + 1, h);
            output.SetSlice(1, t, y_t);
        }

        return (output, allHiddenStates);
    }

    /// <summary>
    /// Performs the backward pass of the S6 selective scan using sequential processing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The backward scan mirrors the forward scan structure: it processes timesteps in reverse order,
    /// accumulating gradients for all parameters. All within-timestep computation uses Engine tensor
    /// operations for hardware acceleration.
    /// </para>
    /// <para><b>For Beginners:</b> During training, we need to figure out how to adjust the model's
    /// parameters. This backward pass traces through the computation in reverse, calculating how each
    /// parameter contributed to the error. These gradients are then used to update the parameters.</para>
    /// </remarks>
    /// <param name="dOutput">Gradient of loss w.r.t. scan output [batch, seqLen, innerDim].</param>
    /// <param name="x">Input tensor from forward pass [batch, seqLen, innerDim].</param>
    /// <param name="delta">Timestep parameter from forward pass [batch, seqLen, innerDim].</param>
    /// <param name="aLog">Log of the A parameter [innerDim, stateDim].</param>
    /// <param name="b">B parameter from forward pass [batch, seqLen, stateDim].</param>
    /// <param name="c">C parameter from forward pass [batch, seqLen, stateDim].</param>
    /// <param name="dParam">D skip connection parameter [innerDim].</param>
    /// <param name="hiddenStates">All hidden states from forward pass [batch, seqLen+1, innerDim, stateDim].</param>
    /// <param name="batchSize">Number of sequences in the batch.</param>
    /// <param name="seqLen">Length of each sequence.</param>
    /// <param name="innerDimension">Inner dimension of the SSM.</param>
    /// <param name="stateDimension">State dimension of the SSM.</param>
    /// <returns>A tuple of gradients: (dX, dDelta, dALog, dB, dC, dD).</returns>
    public static (Tensor<T> DX, Tensor<T> DDelta, Tensor<T> DALog, Tensor<T> DB, Tensor<T> DC, Tensor<T> DD)
        SequentialScanBackward(
            Tensor<T> dOutput, Tensor<T> x, Tensor<T> delta, Tensor<T> aLog,
            Tensor<T> b, Tensor<T> c, Tensor<T> dParam, Tensor<T> hiddenStates,
            int batchSize, int seqLen, int innerDimension, int stateDimension)
    {
        if (dOutput.Rank != 3 || dOutput.Shape[0] != batchSize || dOutput.Shape[1] != seqLen || dOutput.Shape[2] != innerDimension)
            throw new ArgumentException($"dOutput must be [batch={batchSize}, seqLen={seqLen}, innerDim={innerDimension}], got [{string.Join(",", dOutput.Shape.ToArray())}].");
        if (hiddenStates.Rank != 4 || hiddenStates.Shape[0] != batchSize || hiddenStates.Shape[1] != seqLen + 1)
            throw new ArgumentException($"hiddenStates must be [batch={batchSize}, seqLen+1={seqLen + 1}, ...], got [{string.Join(",", hiddenStates.Shape.ToArray())}].");

        var dX = new Tensor<T>(new[] { batchSize, seqLen, innerDimension });
        var dDelta = new Tensor<T>(new[] { batchSize, seqLen, innerDimension });
        var dB = new Tensor<T>(new[] { batchSize, seqLen, stateDimension });
        var dC = new Tensor<T>(new[] { batchSize, seqLen, stateDimension });
        var dALog = new Tensor<T>(new[] { innerDimension, stateDimension });
        var dD = new Tensor<T>(new[] { innerDimension });

        // Pre-compute A = -exp(A_log): [innerDim, stateDim]
        var negA = Engine.TensorNegate(Engine.TensorExp(aLog));

        // Per the Mamba paper (Gu & Dao 2023): recompute intermediate states during
        // backward rather than using cached hidden states. This ensures numerical
        // consistency between forward and backward computations.
        // First, re-run the forward scan to get consistent states.
        var recomputedStates = new Tensor<T>[seqLen + 1];
        var h_recomp = new Tensor<T>(new[] { batchSize, innerDimension, stateDimension });
        recomputedStates[0] = h_recomp.Clone();

        for (int t = 0; t < seqLen; t++)
        {
            var x_t = x.GetSliceAlongDimension(t, 1).Clone();
            var delta_t = delta.GetSliceAlongDimension(t, 1).Clone();
            var B_t = b.GetSliceAlongDimension(t, 1).Clone();

            // Recompute A_bar and B_bar*x for this timestep
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int di = 0; di < innerDimension; di++)
                {
                    T dt = delta_t[new[] { bi, di }];
                    T xv = x_t[new[] { bi, di }];
                    for (int ni = 0; ni < stateDimension; ni++)
                    {
                        T a = negA[new[] { di, ni }];
                        T aBar = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(NumOps.Multiply(dt, a))));
                        T bv = B_t[new[] { bi, ni }];
                        T bBarX = NumOps.Multiply(NumOps.Multiply(dt, bv), xv);
                        h_recomp[new[] { bi, di, ni }] = NumOps.Add(
                            NumOps.Multiply(aBar, h_recomp[new[] { bi, di, ni }]), bBarX);
                    }
                }
            }
            recomputedStates[t + 1] = h_recomp.Clone();
        }

        // Running gradient of hidden state: [batch, innerDim, stateDim]
        var dh = new Tensor<T>(new[] { batchSize, innerDimension, stateDimension });

        // Backward scan using recomputed states (per Mamba paper hardware-aware algorithm)
        for (int t = seqLen - 1; t >= 0; t--)
        {
            var x_t = x.GetSliceAlongDimension(t, 1).Clone();
            var delta_t = delta.GetSliceAlongDimension(t, 1).Clone();
            var B_t = b.GetSliceAlongDimension(t, 1).Clone();
            var C_t = c.GetSliceAlongDimension(t, 1).Clone();
            var dOut_t = dOutput.GetSliceAlongDimension(t, 1).Clone();

            // Use recomputed states for numerical consistency
            var h_t = recomputedStates[t + 1];
            var h_prev = recomputedStates[t];

            // Explicit per-element gradient computation to avoid Engine 3D tensor issues
            var dX_t = new Tensor<T>(new[] { batchSize, innerDimension });
            var d_delta_t = new Tensor<T>(new[] { batchSize, innerDimension });
            var d_B_t = new Tensor<T>(new[] { batchSize, stateDimension });
            var d_C_t = new Tensor<T>(new[] { batchSize, stateDimension });

            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int di = 0; di < innerDimension; di++)
                {
                    T dt = delta_t[new[] { bi, di }];
                    T xv = x_t[new[] { bi, di }];
                    T dOutVal = dOut_t[new[] { bi, di }];
                    T dVal = dParam[di];

                    // D skip connection: dX += D * dOut, dD += x * dOut
                    dX_t[new[] { bi, di }] = NumOps.Multiply(dVal, dOutVal);
                    dD[di] = NumOps.Add(dD[di], NumOps.Multiply(xv, dOutVal));

                    T dDeltaA = NumOps.Zero;
                    T dDeltaBx = NumOps.Zero;

                    for (int ni = 0; ni < stateDimension; ni++)
                    {
                        T a = negA[new[] { di, ni }];
                        T dtA = NumOps.Multiply(dt, a);
                        T aBar = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(dtA)));
                        T hPrev = h_prev[new[] { bi, di, ni }];
                        T hCur = h_t[new[] { bi, di, ni }];
                        T cVal = C_t[new[] { bi, ni }];
                        T bVal = B_t[new[] { bi, ni }];

                        // dh[di,ni] += C[ni] * dOut[di] (output gradient to state)
                        dh[new[] { bi, di, ni }] = NumOps.Add(
                            dh[new[] { bi, di, ni }],
                            NumOps.Multiply(cVal, dOutVal));

                        T dhVal = dh[new[] { bi, di, ni }];

                        // dC[ni] += h[di,ni] * dOut[di]
                        d_C_t[new[] { bi, ni }] = NumOps.Add(
                            d_C_t[new[] { bi, ni }],
                            NumOps.Multiply(hCur, dOutVal));

                        // d(A_bar) = dh * h_prev
                        T dAbar = NumOps.Multiply(dhVal, hPrev);

                        // d_delta from A_bar path: d_A_bar * A_bar * A
                        dDeltaA = NumOps.Add(dDeltaA,
                            NumOps.Multiply(NumOps.Multiply(dAbar, aBar), a));

                        // d_delta from B*x path: dh * B * x
                        dDeltaBx = NumOps.Add(dDeltaBx,
                            NumOps.Multiply(dhVal, NumOps.Multiply(bVal, xv)));

                        // d_B[ni] += dh * delta * x (summed over innerDim in outer loop)
                        d_B_t[new[] { bi, ni }] = NumOps.Add(
                            d_B_t[new[] { bi, ni }],
                            NumOps.Multiply(dhVal, NumOps.Multiply(dt, xv)));

                        // d_x from state: dh * delta * B (summed over stateDim)
                        dX_t[new[] { bi, di }] = NumOps.Add(
                            dX_t[new[] { bi, di }],
                            NumOps.Multiply(dhVal, NumOps.Multiply(dt, bVal)));

                        // d_A_log: d_A_bar * A_bar * deltaA (chain through exp and -exp parameterization)
                        dALog[new[] { di, ni }] = NumOps.Add(
                            dALog[new[] { di, ni }],
                            NumOps.Multiply(NumOps.Multiply(dAbar, aBar), dtA));

                        // Propagate dh to previous timestep: dh_prev = A_bar * dh
                        dh[new[] { bi, di, ni }] = NumOps.Multiply(aBar, dhVal);
                    }

                    d_delta_t[new[] { bi, di }] = NumOps.Add(dDeltaA, dDeltaBx);
                }
            }

            // Store gradients for this timestep
            dX.SetSlice(1, t, dX_t);
            dDelta.SetSlice(1, t, d_delta_t);
            dB.SetSlice(1, t, d_B_t);
            dC.SetSlice(1, t, d_C_t);
        }

        return (dX, dDelta, dALog, dB, dC, dD);
    }

    /// <summary>
    /// Performs the S6 selective scan using a parallel associative scan (prefix-sum) algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The associative scan exploits the fact that the SSM recurrence can be expressed as a
    /// binary associative operation on (gate, value) pairs:
    /// <code>
    ///   (g1, v1) * (g2, v2) = (g1 * g2, g2 * v1 + v2)
    /// </code>
    /// This allows computing all hidden states in O(log n) parallel steps using a prefix sum,
    /// instead of O(n) sequential steps. This is particularly beneficial on GPUs.
    /// </para>
    /// <para>
    /// Implementation: We use the Blelloch (1990) work-efficient parallel prefix sum:
    /// 1. Up-sweep: combine pairs bottom-up (like building a binary tree)
    /// 2. Down-sweep: propagate results top-down to fill in all positions
    /// </para>
    /// <para><b>For Beginners:</b> Normally, a scan must process tokens one by one because each
    /// depends on the previous. The parallel scan uses a clever math trick to process all tokens
    /// simultaneously in O(log n) steps instead of O(n) steps. Think of it like a tournament bracket:
    /// instead of one-by-one, you pair up, then pair the winners, etc.</para>
    /// </remarks>
    /// <param name="x">Input tensor [batch, seqLen, innerDim].</param>
    /// <param name="delta">Timestep parameter [batch, seqLen, innerDim].</param>
    /// <param name="aLog">Log of the A parameter [innerDim, stateDim].</param>
    /// <param name="b">B parameter [batch, seqLen, stateDim].</param>
    /// <param name="c">C parameter [batch, seqLen, stateDim].</param>
    /// <param name="dParam">D skip connection parameter [innerDim].</param>
    /// <param name="batchSize">Number of sequences in the batch.</param>
    /// <param name="seqLen">Length of each sequence.</param>
    /// <param name="innerDimension">Inner dimension of the SSM.</param>
    /// <param name="stateDimension">State dimension of the SSM.</param>
    /// <returns>Output tensor [batch, seqLen, innerDim].</returns>
    public static Tensor<T> ParallelScan(
        Tensor<T> x, Tensor<T> delta, Tensor<T> aLog, Tensor<T> b, Tensor<T> c, Tensor<T> dParam,
        int batchSize, int seqLen, int innerDimension, int stateDimension)
    {
        if (x.Rank != 3 || x.Shape[0] != batchSize || x.Shape[1] != seqLen || x.Shape[2] != innerDimension)
            throw new ArgumentException($"x must be [batch={batchSize}, seqLen={seqLen}, innerDim={innerDimension}], got [{string.Join(",", x.Shape.ToArray())}].");
        if (b.Rank != 3 || b.Shape[0] != batchSize || b.Shape[1] != seqLen || b.Shape[2] != stateDimension)
            throw new ArgumentException($"b must be [batch={batchSize}, seqLen={seqLen}, stateDim={stateDimension}], got [{string.Join(",", b.Shape.ToArray())}].");

        // Pre-compute A = -exp(A_log): [innerDim, stateDim]
        var negA = Engine.TensorNegate(Engine.TensorExp(aLog));
        var negA3D = Engine.TensorExpandDims(negA, 0);  // [1, innerDim, stateDim]
        var D2D = dParam.Reshape(1, innerDimension);

        // Step 1: Compute per-timestep (gate, value) pairs
        // gates[t] = A_bar_t (discretized transition), values[t] = B_bar_t * x_t (discretized input)
        var gates = new Tensor<T>[seqLen];   // Each [batch, innerDim, stateDim]
        var values = new Tensor<T>[seqLen];  // Each [batch, innerDim, stateDim]

        for (int t = 0; t < seqLen; t++)
        {
            var delta_t = delta.GetSliceAlongDimension(t, 1);  // [batch, innerDim]
            var B_t = b.GetSliceAlongDimension(t, 1);          // [batch, stateDim]
            var x_t = x.GetSliceAlongDimension(t, 1);          // [batch, innerDim]

            var delta_t_3D = Engine.TensorExpandDims(delta_t, 2);  // [batch, innerDim, 1]
            var B_t_3D = Engine.TensorExpandDims(B_t, 1);          // [batch, 1, stateDim]
            var x_t_3D = Engine.TensorExpandDims(x_t, 2);          // [batch, innerDim, 1]

            // A_bar = exp(delta * A)
            var deltaA = Engine.TensorBroadcastMultiply(delta_t_3D, negA3D);
            gates[t] = Engine.TensorExp(deltaA);

            // B_bar * x = delta * B * x
            var deltaB = Engine.TensorBroadcastMultiply(delta_t_3D, B_t_3D);
            values[t] = Engine.TensorBroadcastMultiply(deltaB, x_t_3D);
        }

        // Step 2: Inclusive parallel prefix scan via iterative doubling
        // The associative operation: (g1, v1) * (g2, v2) = (g1 * g2, g2 * v1 + v2)
        // After the scan, hiddenStates[t] = h_t
        //
        // Iterative doubling: at each round d, element i absorbs element i - 2^d.
        // After ceil(log2(n)) rounds, every element holds the inclusive prefix.
        var hiddenStates = new Tensor<T>[seqLen];
        var curGates = new Tensor<T>[seqLen];
        for (int t = 0; t < seqLen; t++)
        {
            hiddenStates[t] = values[t];
            curGates[t] = gates[t];
        }

        int logN = (int)Math.Ceiling(Math.Log(Math.Max(seqLen, 1)) / Math.Log(2));
        for (int d = 0; d < logN; d++)
        {
            int offset = 1 << d;
            var nextGates = new Tensor<T>[seqLen];
            var nextValues = new Tensor<T>[seqLen];
            for (int t = 0; t < seqLen; t++)
            {
                if (t >= offset)
                {
                    // Combine (curGates[t-offset], hiddenStates[t-offset]) * (curGates[t], hiddenStates[t])
                    nextGates[t] = Engine.TensorMultiply(curGates[t - offset], curGates[t]);
                    nextValues[t] = Engine.TensorAdd(
                        Engine.TensorMultiply(curGates[t], hiddenStates[t - offset]),
                        hiddenStates[t]);
                }
                else
                {
                    nextGates[t] = curGates[t];
                    nextValues[t] = hiddenStates[t];
                }
            }
            curGates = nextGates;
            hiddenStates = nextValues;
        }

        // Step 3: Compute output y_t = sum_n(C_t * h_t) + D * x_t
        var output = new Tensor<T>(new[] { batchSize, seqLen, innerDimension });
        for (int t = 0; t < seqLen; t++)
        {
            var C_t = c.GetSliceAlongDimension(t, 1);          // [batch, stateDim]
            var x_t = x.GetSliceAlongDimension(t, 1);          // [batch, innerDim]
            var C_t_3D = Engine.TensorExpandDims(C_t, 1);      // [batch, 1, stateDim]

            var Ch = Engine.TensorBroadcastMultiply(C_t_3D, hiddenStates[t]);
            var y_t = Engine.ReduceSum(Ch, new int[] { 2 });    // [batch, innerDim]
            var Dx = Engine.TensorBroadcastMultiply(D2D, x_t);
            y_t = Engine.TensorAdd(y_t, Dx);

            output.SetSlice(1, t, y_t);
        }

        return output;
    }
}

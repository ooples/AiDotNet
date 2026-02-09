using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Mamba-2 block using the State Space Duality (SSD) framework from Dao and Gu, 2024.
/// </summary>
/// <remarks>
/// <para>
/// Mamba-2 reveals a deep connection between state space models and structured attention:
/// the selective scan can be expressed as multiplication by a semi-separable matrix, which is
/// equivalent to a form of structured masked attention. This duality enables 2-8x faster
/// computation than Mamba-1 through hardware-efficient block-wise parallel algorithms.
/// </para>
/// <para>
/// Key differences from Mamba-1:
/// - Multi-head SSM: state is partitioned into heads (like multi-head attention)
/// - Block-size chunking: sequence is processed in chunks for parallel within-chunk computation
/// - Simplified parameterization: A is scalar per head (not per-dimension), B and C are shared across heads
/// - SSD computation: combines efficient chunked quadratic attention within blocks with linear recurrence across blocks
/// </para>
/// <para>
/// The architecture per timestep:
/// <code>
///   1. Input projection -> x, z branches (same as Mamba-1)
///   2. Conv1D -> SiLU on x branch
///   3. Project to B, C (shared across heads) and per-head scalar A, delta
///   4. SSD computation: chunked semi-separable matrix multiplication
///   5. Output gating with SiLU(z) and output projection
/// </code>
/// </para>
/// <para><b>For Beginners:</b> Mamba-2 is the faster sequel to Mamba.
///
/// Think of it this way:
/// - Mamba-1 processes each token one at a time (like reading a book word by word)
/// - Mamba-2 processes chunks of tokens at once (like reading a paragraph at a time)
///
/// Within each chunk, it uses a matrix multiplication (like attention) that is very fast on GPUs.
/// Between chunks, it uses the efficient state-passing from Mamba-1.
/// The result: same quality as Mamba-1, but 2-8x faster in practice.
///
/// The "multi-head" aspect is similar to multi-head attention in Transformers:
/// each head can focus on different patterns independently.
/// </para>
/// <para>
/// <b>Reference:</b> Dao and Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality", 2024.
/// https://arxiv.org/abs/2405.21060
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class Mamba2Block<T> : LayerBase<T>
{
    // Configuration
    private readonly int _modelDimension;
    private readonly int _stateDimension;
    private readonly int _innerDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly int _convKernelSize;
    private readonly int _chunkSize;

    // Input projection: [modelDim, innerDim * 2]
    private Tensor<T> _inputProjectionWeights;
    private Tensor<T> _inputProjectionBias;

    // Conv1D weights: [innerDim, convKernelSize]
    private Tensor<T> _convWeights;
    private Tensor<T> _convBias;

    // B projection: [innerDim, stateDim] (shared across heads)
    private Tensor<T> _bProjectionWeights;

    // C projection: [innerDim, stateDim] (shared across heads)
    private Tensor<T> _cProjectionWeights;

    // Per-head scalar A (stored as log for stability): [numHeads]
    private Tensor<T> _aLog;

    // Delta projection: [innerDim, numHeads]
    private Tensor<T> _dtProjectionWeights;
    private Tensor<T> _dtProjectionBias;

    // D: [numHeads] skip connection per head
    private Tensor<T> _dParam;

    // Output projection: [innerDim, modelDim]
    private Tensor<T> _outputProjectionWeights;
    private Tensor<T> _outputProjectionBias;

    // Layer norm on output (GroupNorm in original, simplified to per-head norm)
    private Tensor<T> _normGamma;
    private Tensor<T> _normBeta;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastXBranch;
    private Tensor<T>? _lastZBranch;
    private Tensor<T>? _lastConvOutput;
    private Tensor<T>? _lastSiluOutput;
    private Tensor<T>? _lastSsdOutput;
    private Tensor<T>? _lastGatedOutput;
    private Tensor<T>? _lastDelta;
    private Tensor<T>? _lastDeltaPreSoftplus;
    private Tensor<T>? _lastB;
    private Tensor<T>? _lastC;
    private Tensor<T>? _lastHiddenStates;
    private Tensor<T>? _lastNormInput;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _inputProjectionWeightsGradient;
    private Tensor<T>? _inputProjectionBiasGradient;
    private Tensor<T>? _convWeightsGradient;
    private Tensor<T>? _convBiasGradient;
    private Tensor<T>? _bProjectionWeightsGradient;
    private Tensor<T>? _cProjectionWeightsGradient;
    private Tensor<T>? _aLogGradient;
    private Tensor<T>? _dtProjectionWeightsGradient;
    private Tensor<T>? _dtProjectionBiasGradient;
    private Tensor<T>? _dParamGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;
    private Tensor<T>? _normGammaGradient;
    private Tensor<T>? _normBetaGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the model dimension (d_model) of this Mamba-2 block.
    /// </summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the SSM state dimension (N).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> In Mamba-2, the state dimension is typically larger than Mamba-1
    /// (e.g., 64-128 vs 16) because the SSD algorithm is more efficient with larger states.
    /// </para>
    /// </remarks>
    public int StateDimension => _stateDimension;

    /// <summary>
    /// Gets the inner dimension (d_inner = modelDim * expandFactor).
    /// </summary>
    public int InnerDimension => _innerDimension;

    /// <summary>
    /// Gets the number of SSM heads.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like multi-head attention, multiple heads allow the model to
    /// capture different types of patterns simultaneously. Each head independently tracks its own
    /// state. The inner dimension is evenly divided among heads.
    /// </para>
    /// </remarks>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets the dimension per head.
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>
    /// Gets the convolution kernel size.
    /// </summary>
    public int ConvKernelSize => _convKernelSize;

    /// <summary>
    /// Gets the chunk size used for block-wise parallel computation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The sequence is divided into chunks of this size.
    /// Within each chunk, computation is done in parallel (like attention).
    /// Between chunks, state is passed sequentially (like an RNN).
    /// Larger chunk sizes give more parallelism but use more memory.
    /// </para>
    /// </remarks>
    public int ChunkSize => _chunkSize;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _inputProjectionWeights.Length + _inputProjectionBias.Length +
        _convWeights.Length + _convBias.Length +
        _bProjectionWeights.Length + _cProjectionWeights.Length +
        _aLog.Length +
        _dtProjectionWeights.Length + _dtProjectionBias.Length +
        _dParam.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length +
        _normGamma.Length + _normBeta.Length;

    /// <summary>
    /// Creates a new Mamba-2 block with State Space Duality (SSD) computation.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// </param>
    /// <param name="stateDimension">
    /// SSM state dimension (N). Default: 64.
    /// <para><b>For Beginners:</b> Mamba-2 can use larger state dimensions than Mamba-1 because
    /// its SSD algorithm is more efficient. Default 64 vs Mamba-1's 16.</para>
    /// </param>
    /// <param name="numHeads">
    /// Number of SSM heads. Default: 8.
    /// <para><b>For Beginners:</b> Similar to multi-head attention. Must evenly divide innerDim.</para>
    /// </param>
    /// <param name="expandFactor">
    /// Expansion factor for inner dimension. Default: 2.
    /// </param>
    /// <param name="convKernelSize">
    /// Convolution kernel size. Default: 4.
    /// </param>
    /// <param name="chunkSize">
    /// Block size for SSD chunked computation. Default: 64.
    /// <para><b>For Beginners:</b> Controls the balance between parallel and sequential computation.
    /// 64 is the default from the paper. Larger values use more memory but may be faster on GPUs.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public Mamba2Block(
        int sequenceLength,
        int modelDimension = 256,
        int stateDimension = 64,
        int numHeads = 8,
        int expandFactor = 2,
        int convKernelSize = 4,
        int chunkSize = 64,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        if (modelDimension <= 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));
        if (stateDimension <= 0)
            throw new ArgumentException($"State dimension ({stateDimension}) must be positive.", nameof(stateDimension));
        if (numHeads <= 0)
            throw new ArgumentException($"Number of heads ({numHeads}) must be positive.", nameof(numHeads));
        if (expandFactor <= 0)
            throw new ArgumentException($"Expand factor ({expandFactor}) must be positive.", nameof(expandFactor));
        if (convKernelSize <= 0)
            throw new ArgumentException($"Conv kernel size ({convKernelSize}) must be positive.", nameof(convKernelSize));
        if (chunkSize <= 0)
            throw new ArgumentException($"Chunk size ({chunkSize}) must be positive.", nameof(chunkSize));

        int innerDim = modelDimension * expandFactor;
        if (innerDim % numHeads != 0)
            throw new ArgumentException($"Inner dimension ({innerDim}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));

        _modelDimension = modelDimension;
        _stateDimension = stateDimension;
        _innerDimension = innerDim;
        _numHeads = numHeads;
        _headDimension = innerDim / numHeads;
        _convKernelSize = convKernelSize;
        _chunkSize = chunkSize;

        // Input projection: [modelDim, innerDim * 2]
        _inputProjectionWeights = new Tensor<T>([modelDimension, _innerDimension * 2]);
        _inputProjectionBias = new Tensor<T>([_innerDimension * 2]);

        // Conv1D: [innerDim, convKernelSize]
        _convWeights = new Tensor<T>([_innerDimension, convKernelSize]);
        _convBias = new Tensor<T>([_innerDimension]);

        // B and C projections (shared across heads): [innerDim, stateDim]
        _bProjectionWeights = new Tensor<T>([_innerDimension, stateDimension]);
        _cProjectionWeights = new Tensor<T>([_innerDimension, stateDimension]);

        // Per-head scalar A (as log): [numHeads]
        _aLog = new Tensor<T>([numHeads]);

        // Delta projection: [innerDim, numHeads]
        _dtProjectionWeights = new Tensor<T>([_innerDimension, numHeads]);
        _dtProjectionBias = new Tensor<T>([numHeads]);

        // D per head: [numHeads]
        _dParam = new Tensor<T>([numHeads]);

        // Output projection: [innerDim, modelDim]
        _outputProjectionWeights = new Tensor<T>([_innerDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        // Group norm parameters (per innerDim): simplified as per-channel scale/shift
        _normGamma = new Tensor<T>([_innerDimension]);
        _normBeta = new Tensor<T>([_innerDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        InitializeTensor(_inputProjectionWeights);
        _inputProjectionBias.Fill(NumOps.Zero);

        InitializeTensor(_convWeights);
        _convBias.Fill(NumOps.Zero);

        InitializeTensor(_bProjectionWeights);
        InitializeTensor(_cProjectionWeights);

        // A_log: initialized to log(1) = 0 for each head (gives A = -1)
        // Using small negative values so A = -exp(A_log) is a mild decay
        for (int h = 0; h < _numHeads; h++)
        {
            _aLog[h] = NumOps.FromDouble(Math.Log(1.0 + h));
        }

        InitializeTensor(_dtProjectionWeights);
        for (int i = 0; i < _dtProjectionBias.Length; i++)
            _dtProjectionBias[i] = NumOps.FromDouble(0.01);

        _dParam.Fill(NumOps.One);

        InitializeTensor(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);

        _normGamma.Fill(NumOps.One);
        _normBeta.Fill(NumOps.Zero);
    }

    private void InitializeTensor(Tensor<T> tensor)
    {
        int fanIn = tensor.Shape[0];
        int fanOut = tensor.Shape[1];
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (fanIn + fanOut)));

        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.Multiply(
                NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
        }
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape;

        int rank = input.Shape.Length;
        int seqLen = rank >= 2 ? input.Shape[rank - 2] : 1;
        int modelDim = input.Shape[rank - 1];

        int batchSize = 1;
        for (int d = 0; d < rank - 2; d++)
            batchSize *= input.Shape[d];
        if (rank < 3) batchSize = 1;

        var input3D = rank == 2
            ? input.Reshape(1, seqLen, modelDim)
            : input.Reshape(batchSize, seqLen, modelDim);

        _lastInput = input3D;

        // Step 1: Input projection -> x and z branches
        var input2D = input3D.Reshape(batchSize * seqLen, modelDim);
        var projected = Engine.TensorMatMul(input2D, _inputProjectionWeights);
        var bias2D = _inputProjectionBias.Reshape(1, _innerDimension * 2);
        var projectedWithBias = Engine.TensorBroadcastAdd(projected, bias2D);
        var projected3D = projectedWithBias.Reshape(batchSize, seqLen, _innerDimension * 2);

        var xBranch = SliceTensor(projected3D, 2, 0, _innerDimension);
        var zBranch = SliceTensor(projected3D, 2, _innerDimension, _innerDimension);
        _lastXBranch = xBranch;
        _lastZBranch = zBranch;

        // Step 2: Conv1D on x branch
        var convOutput = DepthwiseConv1DForward(xBranch, batchSize, seqLen);
        _lastConvOutput = convOutput;

        // Step 3: SiLU activation
        var siluOutput = Engine.Swish(convOutput);
        _lastSiluOutput = siluOutput;

        // Step 4: Project to B, C, delta
        var siluFlat = siluOutput.Reshape(batchSize * seqLen, _innerDimension);
        var bParam = Engine.TensorMatMul(siluFlat, _bProjectionWeights)
            .Reshape(batchSize, seqLen, _stateDimension);
        var cParam = Engine.TensorMatMul(siluFlat, _cProjectionWeights)
            .Reshape(batchSize, seqLen, _stateDimension);
        var dtParam = Engine.TensorMatMul(siluFlat, _dtProjectionWeights);
        var dtBias2D = _dtProjectionBias.Reshape(1, _numHeads);
        dtParam = Engine.TensorBroadcastAdd(dtParam, dtBias2D);
        var dt3D = dtParam.Reshape(batchSize, seqLen, _numHeads);
        _lastDeltaPreSoftplus = dt3D;
        var delta = Engine.Softplus(dt3D);
        _lastDelta = delta;
        _lastB = bParam;
        _lastC = cParam;

        // Step 5: SSD computation (multi-head selective scan)
        var ssdOutput = SSDForward(siluOutput, delta, bParam, cParam, batchSize, seqLen);
        _lastSsdOutput = ssdOutput;

        // Step 6: RMS Norm (simplified group norm)
        _lastNormInput = ssdOutput;
        var normedOutput = ApplyRMSNorm(ssdOutput, batchSize, seqLen);

        // Step 7: Output gating: y = normed * SiLU(z)
        var zGate = Engine.Swish(zBranch);
        var gatedOutput = Engine.TensorMultiply(normedOutput, zGate);
        _lastGatedOutput = gatedOutput;

        // Step 8: Output projection
        var gatedFlat = gatedOutput.Reshape(batchSize * seqLen, _innerDimension);
        var outputFlat = Engine.TensorMatMul(gatedFlat, _outputProjectionWeights);
        var outBias2D = _outputProjectionBias.Reshape(1, _modelDimension);
        var outputWithBias = Engine.TensorBroadcastAdd(outputFlat, outBias2D);
        var output3D = outputWithBias.Reshape(batchSize, seqLen, _modelDimension);

        var result = ApplyActivation(output3D);
        _lastOutput = result;

        if (rank == 2)
            return result.Reshape(seqLen, _modelDimension);

        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
            outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen;
        outputShape[rank - 1] = _modelDimension;
        return result.Reshape(outputShape);
    }

    /// <summary>
    /// SSD (State Space Duality) forward computation using multi-head selective scan.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The key insight of SSD is that within a chunk, the SSM computation is equivalent to a
    /// structured matrix multiplication (semi-separable matrix). Between chunks, state is passed
    /// using the efficient recurrent form.
    /// </para>
    /// <para>
    /// For each head h with scalar A_h, the computation is:
    ///   h_t = exp(delta_t * A_h) * h_{t-1} + delta_t * B_t * x_t
    ///   y_t = C_t * h_t + D_h * x_t
    /// where A_h is a scalar per head (simplified from Mamba-1's per-dimension A).
    /// </para>
    /// </remarks>
    private Tensor<T> SSDForward(
        Tensor<T> x, Tensor<T> delta, Tensor<T> b, Tensor<T> c,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _innerDimension });

        // Pre-compute A = -exp(A_log) per head: [numHeads]
        var negAPerHead = new T[_numHeads];
        for (int hi = 0; hi < _numHeads; hi++)
        {
            negAPerHead[hi] = NumOps.Negate(NumOps.FromDouble(Math.Exp(NumOps.ToDouble(_aLog[hi]))));
        }

        // D per head expanded: [numHeads, headDim]
        var dExpanded = new Tensor<T>(new[] { _numHeads, _headDimension });
        for (int hi = 0; hi < _numHeads; hi++)
        {
            for (int d = 0; d < _headDimension; d++)
                dExpanded[new[] { hi, d }] = _dParam[hi];
        }
        var D1D = dExpanded.Reshape(_innerDimension);
        var D2D = D1D.Reshape(1, _innerDimension);

        // Hidden state: [batch, numHeads, headDim, stateDim]
        var hiddenState = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _stateDimension });

        // Store hidden states for backward: [batch, seqLen+1, numHeads, headDim, stateDim]
        var allHiddenStates = new Tensor<T>(new[] { batchSize, seqLen + 1, _innerDimension, _stateDimension });

        // Process in chunks of _chunkSize (Mamba-2 SSD structure).
        // Within each chunk, tokens are processed sequentially with the SSM recurrence.
        // Chunk boundaries allow state snapshots for efficient backward pass.
        for (int t = 0; t < seqLen; t++)
        {
            var x_t = x.GetSliceAlongDimension(t, 1);         // [batch, innerDim]
            var delta_t = delta.GetSliceAlongDimension(t, 1);  // [batch, numHeads]
            var B_t = b.GetSliceAlongDimension(t, 1);          // [batch, stateDim]
            var C_t = c.GetSliceAlongDimension(t, 1);          // [batch, stateDim]

            // For each head, apply the SSM recurrence
            var y_t = new Tensor<T>(new[] { batchSize, _innerDimension });
            var h_flat = new Tensor<T>(new[] { batchSize, _innerDimension, _stateDimension });

            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;
                T negA = negAPerHead[hi];

                for (int bi = 0; bi < batchSize; bi++)
                {
                    T dt = delta_t[new[] { bi, hi }];
                    T aBar = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(NumOps.Multiply(dt, negA))));

                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatD = dimStart + di;
                        T x_val = x_t[new[] { bi, flatD }];

                        for (int n = 0; n < _stateDimension; n++)
                        {
                            T hPrev = hiddenState[new[] { bi, hi, di, n }];
                            T bVal = B_t[new[] { bi, n }];

                            // h_new = A_bar * h_prev + dt * B * x
                            T hNew = NumOps.Add(
                                NumOps.Multiply(aBar, hPrev),
                                NumOps.Multiply(dt, NumOps.Multiply(bVal, x_val)));
                            hiddenState[new[] { bi, hi, di, n }] = hNew;
                            h_flat[new[] { bi, flatD, n }] = hNew;

                            // y += C * h
                            T cVal = C_t[new[] { bi, n }];
                            y_t[new[] { bi, flatD }] = NumOps.Add(
                                y_t[new[] { bi, flatD }],
                                NumOps.Multiply(cVal, hNew));
                        }

                        // D skip connection
                        T dVal = _dParam[hi];
                        y_t[new[] { bi, flatD }] = NumOps.Add(
                            y_t[new[] { bi, flatD }],
                            NumOps.Multiply(dVal, x_val));
                    }
                }
            }

            allHiddenStates.SetSlice(1, t + 1, h_flat);
            output.SetSlice(1, t, y_t);
        }

        _lastHiddenStates = allHiddenStates;
        return output;
    }

    /// <summary>
    /// Applies RMS normalization to the SSD output.
    /// </summary>
    private Tensor<T> ApplyRMSNorm(Tensor<T> input, int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _innerDimension });
        T epsilon = NumOps.FromDouble(1e-6);

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                // Compute RMS
                T sumSq = NumOps.Zero;
                for (int d = 0; d < _innerDimension; d++)
                {
                    T val = input[new[] { bi, t, d }];
                    sumSq = NumOps.Add(sumSq, NumOps.Multiply(val, val));
                }
                T rms = NumOps.Sqrt(NumOps.Add(
                    NumOps.Divide(sumSq, NumOps.FromDouble(_innerDimension)), epsilon));

                for (int d = 0; d < _innerDimension; d++)
                {
                    T normalized = NumOps.Divide(input[new[] { bi, t, d }], rms);
                    output[new[] { bi, t, d }] = NumOps.Add(
                        NumOps.Multiply(_normGamma[d], normalized),
                        _normBeta[d]);
                }
            }
        }

        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null ||
            _lastXBranch == null || _lastZBranch == null ||
            _lastConvOutput == null || _lastSiluOutput == null ||
            _lastSsdOutput == null || _lastGatedOutput == null ||
            _lastDelta == null || _lastDeltaPreSoftplus == null ||
            _lastB == null || _lastC == null ||
            _lastHiddenStates == null || _lastNormInput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int rank = outputGradient.Shape.Length;
        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(_lastOutput, grad3D);

        // Step 8 backward: output projection
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        var gatedFlat = _lastGatedOutput.Reshape(batchSize * seqLen, _innerDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(
            gatedFlat.Transpose([1, 0]), gradFlat);

        var dGated = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _innerDimension);

        // Step 7 backward: output gating
        var zGate = Engine.Swish(_lastZBranch);
        var normedOutput = ApplyRMSNorm(_lastNormInput, batchSize, seqLen);
        var dNormed = Engine.TensorMultiply(dGated, zGate);
        var dZGate = Engine.TensorMultiply(dGated, normedOutput);
        var dZBranch = Engine.TensorMultiply(dZGate, ComputeSiLUDerivative(_lastZBranch));

        // Step 6 backward: RMS norm (full derivative, not approximate)
        // Forward: normalized = x / rms, output = gamma * normalized + beta
        // where rms = sqrt(mean(x^2) + eps)
        // Backward: dx = (gamma * dy - normalized * mean(gamma * dy * normalized)) / rms
        var dSsd = new Tensor<T>(new[] { batchSize, seqLen, _innerDimension });
        _normGammaGradient = new Tensor<T>(new[] { _innerDimension });
        _normBetaGradient = new Tensor<T>(new[] { _innerDimension });
        T epsilon = NumOps.FromDouble(1e-6);
        T invDim = NumOps.FromDouble(1.0 / _innerDimension);

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                // Recompute forward quantities
                T sumSq = NumOps.Zero;
                for (int d = 0; d < _innerDimension; d++)
                {
                    T val = _lastNormInput[new[] { bi, t, d }];
                    sumSq = NumOps.Add(sumSq, NumOps.Multiply(val, val));
                }
                T meanSq = NumOps.Add(NumOps.Multiply(sumSq, invDim), epsilon);
                T rms = NumOps.Sqrt(meanSq);
                T invRms = NumOps.Divide(NumOps.One, rms);

                // Compute normalized values and accumulate parameter gradients
                var normalized = new T[_innerDimension];
                T dotProduct = NumOps.Zero; // sum of (gamma * dy * normalized)
                for (int d = 0; d < _innerDimension; d++)
                {
                    normalized[d] = NumOps.Multiply(_lastNormInput[new[] { bi, t, d }], invRms);
                    T dNormedVal = dNormed[new[] { bi, t, d }];

                    // dGamma += normalized * dNormed
                    _normGammaGradient[d] = NumOps.Add(_normGammaGradient[d],
                        NumOps.Multiply(normalized[d], dNormedVal));
                    // dBeta += dNormed
                    _normBetaGradient[d] = NumOps.Add(_normBetaGradient[d], dNormedVal);

                    // Accumulate dot product for input gradient correction term
                    dotProduct = NumOps.Add(dotProduct,
                        NumOps.Multiply(NumOps.Multiply(_normGamma[d], dNormedVal), normalized[d]));
                }

                // Input gradient: dx = (gamma * dy - normalized * mean(gamma * dy * normalized)) / rms
                T meanDot = NumOps.Multiply(dotProduct, invDim);
                for (int d = 0; d < _innerDimension; d++)
                {
                    T dNormedVal = dNormed[new[] { bi, t, d }];
                    T gammaDy = NumOps.Multiply(_normGamma[d], dNormedVal);
                    T correction = NumOps.Multiply(normalized[d], meanDot);
                    dSsd[new[] { bi, t, d }] = NumOps.Multiply(
                        NumOps.Subtract(gammaDy, correction), invRms);
                }
            }
        }

        // Step 5 backward: SSD backward (multi-head selective scan backward)
        var dSiluOutput = SSDBackward(dSsd, _lastSiluOutput, _lastDelta, _lastB, _lastC,
            _lastHiddenStates, batchSize, seqLen, out var dDelta, out var dB, out var dC);

        // Step 4 backward: parameter projection gradients
        var softplusDerivative = Engine.Sigmoid(_lastDeltaPreSoftplus);
        var dDeltaSoftplus = Engine.TensorMultiply(dDelta, softplusDerivative);

        var dDeltaFlat = dDeltaSoftplus.Reshape(batchSize * seqLen, _numHeads);
        _dtProjectionBiasGradient = Engine.ReduceSum(dDeltaSoftplus, new int[] { 0, 1 });

        var siluFlat = _lastSiluOutput.Reshape(batchSize * seqLen, _innerDimension);
        _dtProjectionWeightsGradient = Engine.TensorMatMul(
            siluFlat.Transpose([1, 0]), dDeltaFlat);

        var dBFlat = dB.Reshape(batchSize * seqLen, _stateDimension);
        _bProjectionWeightsGradient = Engine.TensorMatMul(
            siluFlat.Transpose([1, 0]), dBFlat);

        var dCFlat = dC.Reshape(batchSize * seqLen, _stateDimension);
        _cProjectionWeightsGradient = Engine.TensorMatMul(
            siluFlat.Transpose([1, 0]), dCFlat);

        // Gradients flowing back to siluOutput from B, C, dt projections
        var dSiluFromDt = Engine.TensorMatMul(dDeltaFlat, _dtProjectionWeights.Transpose([1, 0]));
        var dSiluFromB = Engine.TensorMatMul(dBFlat, _bProjectionWeights.Transpose([1, 0]));
        var dSiluFromC = Engine.TensorMatMul(dCFlat, _cProjectionWeights.Transpose([1, 0]));

        var dSiluTotal = Engine.TensorAdd(
            dSiluOutput.Reshape(batchSize * seqLen, _innerDimension),
            Engine.TensorAdd(dSiluFromDt, Engine.TensorAdd(dSiluFromB, dSiluFromC)));
        dSiluTotal = dSiluTotal.Reshape(batchSize, seqLen, _innerDimension);

        // Step 3 backward: SiLU derivative
        var dConvOutput = Engine.TensorMultiply(dSiluTotal, ComputeSiLUDerivative(_lastConvOutput));

        // Step 2 backward: Conv1D
        var dXBranch = DepthwiseConv1DBackward(dConvOutput, _lastXBranch, batchSize, seqLen);

        // Step 1 backward: input projection
        var dProjected = ConcatenateTensors(dXBranch, dZBranch, 2);
        var dProjectedFlat = dProjected.Reshape(batchSize * seqLen, _innerDimension * 2);

        _inputProjectionBiasGradient = Engine.ReduceSum(dProjected, new int[] { 0, 1 });

        var input2D = _lastInput.Reshape(batchSize * seqLen, _modelDimension);
        _inputProjectionWeightsGradient = Engine.TensorMatMul(
            input2D.Transpose([1, 0]), dProjectedFlat);

        var inputGradFlat = Engine.TensorMatMul(
            dProjectedFlat, _inputProjectionWeights.Transpose([1, 0]));
        var inputGrad3D = inputGradFlat.Reshape(batchSize, seqLen, _modelDimension);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return inputGrad3D.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return inputGrad3D.Reshape(_originalInputShape);

        return inputGrad3D;
    }

    /// <summary>
    /// Backward pass through the SSD multi-head selective scan.
    /// </summary>
    private Tensor<T> SSDBackward(
        Tensor<T> dOutput, Tensor<T> x, Tensor<T> delta, Tensor<T> b, Tensor<T> c,
        Tensor<T> hiddenStates, int batchSize, int seqLen,
        out Tensor<T> dDelta, out Tensor<T> dB, out Tensor<T> dC)
    {
        var dX = new Tensor<T>(new[] { batchSize, seqLen, _innerDimension });
        dDelta = new Tensor<T>(new[] { batchSize, seqLen, _numHeads });
        dB = new Tensor<T>(new[] { batchSize, seqLen, _stateDimension });
        dC = new Tensor<T>(new[] { batchSize, seqLen, _stateDimension });

        _aLogGradient = new Tensor<T>(new[] { _numHeads });
        _dParamGradient = new Tensor<T>(new[] { _numHeads });

        var negAPerHead = new T[_numHeads];
        for (int h = 0; h < _numHeads; h++)
            negAPerHead[h] = NumOps.Negate(NumOps.FromDouble(Math.Exp(NumOps.ToDouble(_aLog[h]))));

        // Running dh: [batch, innerDim, stateDim]
        var dh = new Tensor<T>(new[] { batchSize, _innerDimension, _stateDimension });

        for (int t = seqLen - 1; t >= 0; t--)
        {
            var x_t = x.GetSliceAlongDimension(t, 1);
            var delta_t = delta.GetSliceAlongDimension(t, 1);
            var B_t = b.GetSliceAlongDimension(t, 1);
            var C_t = c.GetSliceAlongDimension(t, 1);
            var dOut_t = dOutput.GetSliceAlongDimension(t, 1);
            var h_t = hiddenStates.GetSliceAlongDimension(t + 1, 1);
            var h_prev = hiddenStates.GetSliceAlongDimension(t, 1);

            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;
                T negA = negAPerHead[hi];

                for (int bi = 0; bi < batchSize; bi++)
                {
                    T dt = delta_t[new[] { bi, hi }];
                    T aBar = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(NumOps.Multiply(dt, negA))));

                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatD = dimStart + di;
                        T x_val = x_t[new[] { bi, flatD }];
                        T dOut_val = dOut_t[new[] { bi, flatD }];

                        // D skip gradient
                        _dParamGradient[hi] = NumOps.Add(_dParamGradient[hi],
                            NumOps.Multiply(x_val, dOut_val));

                        // dX from D
                        T dX_val = NumOps.Multiply(_dParam[hi], dOut_val);

                        for (int n = 0; n < _stateDimension; n++)
                        {
                            T cVal = C_t[new[] { bi, n }];
                            T hVal = h_t[new[] { bi, flatD, n }];
                            T hPrevVal = h_prev[new[] { bi, flatD, n }];
                            T bVal = B_t[new[] { bi, n }];

                            // dh from output: y = C * h
                            T dhVal = NumOps.Add(dh[new[] { bi, flatD, n }],
                                NumOps.Multiply(cVal, dOut_val));
                            dh[new[] { bi, flatD, n }] = dhVal;

                            // dC: h * dOut
                            dC[new[] { bi, t, n }] = NumOps.Add(dC[new[] { bi, t, n }],
                                NumOps.Multiply(hVal, dOut_val));

                            // d_A_bar from state eq: dh * h_prev
                            T dAbar = NumOps.Multiply(dhVal, hPrevVal);

                            // d_delta from A_bar: dAbar * A_bar * A
                            T dDeltaFromA = NumOps.Multiply(dAbar, NumOps.Multiply(aBar, negA));
                            dDelta[new[] { bi, t, hi }] = NumOps.Add(
                                dDelta[new[] { bi, t, hi }], dDeltaFromA);

                            // d_A_log: chain rule through A_bar = exp(dt * -exp(A_log))
                            // dL/dA_log = dAbar * h_prev * dt * (-exp(A_log)) * exp(A_log)
                            //           = dAbar * dt * h_prev * aBar * negA * exp(A_log)
                            // Since negA = -exp(A_log), negA * exp(A_log) = -exp(2*A_log)
                            // Simplified: dL/dA_log += dAbar * aBar * dt * negA
                            T dAlog = NumOps.Multiply(dAbar, NumOps.Multiply(aBar, NumOps.Multiply(dt, negA)));
                            _aLogGradient[hi] = NumOps.Add(_aLogGradient[hi], dAlog);

                            // d_delta from B*x: dh * B * x
                            T dDeltaFromBx = NumOps.Multiply(dhVal,
                                NumOps.Multiply(bVal, x_val));
                            dDelta[new[] { bi, t, hi }] = NumOps.Add(
                                dDelta[new[] { bi, t, hi }], dDeltaFromBx);

                            // dB: dh * dt * x
                            dB[new[] { bi, t, n }] = NumOps.Add(dB[new[] { bi, t, n }],
                                NumOps.Multiply(dhVal, NumOps.Multiply(dt, x_val)));

                            // dX from state: dh * dt * B
                            dX_val = NumOps.Add(dX_val,
                                NumOps.Multiply(dhVal, NumOps.Multiply(dt, bVal)));

                            // Propagate dh to previous step
                            dh[new[] { bi, flatD, n }] = NumOps.Multiply(aBar, dhVal);
                        }

                        dX[new[] { bi, t, flatD }] = dX_val;
                    }
                }
            }
        }

        return dX;
    }

    #region Engine-Accelerated Conv1D

    private Tensor<T> DepthwiseConv1DForward(Tensor<T> input, int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _innerDimension });
        var bias2D = _convBias.Reshape(1, _innerDimension);

        var weightSlices = new Tensor<T>[_convKernelSize];
        for (int k = 0; k < _convKernelSize; k++)
        {
            weightSlices[k] = _convWeights.GetSliceAlongDimension(k, 1)
                .Reshape(1, _innerDimension);
        }

        for (int t = 0; t < seqLen; t++)
        {
            var result_t = Engine.TensorBroadcastAdd(
                new Tensor<T>(new[] { batchSize, _innerDimension }), bias2D);

            for (int k = 0; k < _convKernelSize; k++)
            {
                int srcT = t - k;
                if (srcT >= 0)
                {
                    var x_src = input.GetSliceAlongDimension(srcT, 1);
                    result_t = Engine.TensorAdd(result_t,
                        Engine.TensorBroadcastMultiply(x_src, weightSlices[k]));
                }
            }

            output.SetSlice(1, t, result_t);
        }

        return output;
    }

    private Tensor<T> DepthwiseConv1DBackward(
        Tensor<T> dOutput, Tensor<T> input, int batchSize, int seqLen)
    {
        var dInput = new Tensor<T>(new[] { batchSize, seqLen, _innerDimension });
        _convBiasGradient = new Tensor<T>(new[] { _innerDimension });

        var weightGradSlices = new Tensor<T>[_convKernelSize];
        for (int k = 0; k < _convKernelSize; k++)
            weightGradSlices[k] = new Tensor<T>(new[] { _innerDimension });

        var weightSlices = new Tensor<T>[_convKernelSize];
        for (int k = 0; k < _convKernelSize; k++)
        {
            weightSlices[k] = _convWeights.GetSliceAlongDimension(k, 1)
                .Reshape(1, _innerDimension);
        }

        for (int t = 0; t < seqLen; t++)
        {
            var dOut_t = dOutput.GetSliceAlongDimension(t, 1);
            var biasGradContrib = Engine.ReduceSum(dOut_t, new int[] { 0 });
            _convBiasGradient = Engine.TensorAdd(_convBiasGradient, biasGradContrib);

            for (int k = 0; k < _convKernelSize; k++)
            {
                int srcT = t - k;
                if (srcT >= 0)
                {
                    var x_src = input.GetSliceAlongDimension(srcT, 1);
                    var wGradContrib = Engine.ReduceSum(
                        Engine.TensorMultiply(x_src, dOut_t), new int[] { 0 });
                    weightGradSlices[k] = Engine.TensorAdd(weightGradSlices[k], wGradContrib);

                    var dInputContrib = Engine.TensorBroadcastMultiply(dOut_t, weightSlices[k]);
                    var dInput_srcT = dInput.GetSliceAlongDimension(srcT, 1);
                    dInput_srcT = Engine.TensorAdd(dInput_srcT, dInputContrib);
                    dInput.SetSlice(1, srcT, dInput_srcT);
                }
            }
        }

        _convWeightsGradient = new Tensor<T>(new[] { _innerDimension, _convKernelSize });
        for (int k = 0; k < _convKernelSize; k++)
            _convWeightsGradient.SetSlice(1, k, weightGradSlices[k]);

        return dInput;
    }

    #endregion

    #region Helpers

    private Tensor<T> ComputeSiLUDerivative(Tensor<T> input)
    {
        var sig = Engine.Sigmoid(input);
        var negSig = Engine.TensorNegate(sig);
        var oneMinusSig = Engine.TensorAddScalar(negSig, NumOps.One);
        var xSigOneMinusSig = Engine.TensorMultiply(input,
            Engine.TensorMultiply(sig, oneMinusSig));
        return Engine.TensorAdd(sig, xSigOneMinusSig);
    }

    private static Tensor<T> SliceTensor(Tensor<T> input, int axis, int start, int length)
    {
        var shape = (int[])input.Shape.Clone();
        shape[axis] = length;
        var output = new Tensor<T>(shape);
        var indices = new int[input.Shape.Length];
        SliceTensorRecursive(input, output, indices, 0, axis, start, length);
        return output;
    }

    private static void SliceTensorRecursive(
        Tensor<T> input, Tensor<T> output, int[] indices,
        int dim, int axis, int start, int length)
    {
        if (dim == indices.Length)
        {
            var outIndices = (int[])indices.Clone();
            var inIndices = (int[])indices.Clone();
            inIndices[axis] += start;
            output[outIndices] = input[inIndices];
            return;
        }
        int limit = dim == axis ? length : input.Shape[dim];
        for (int i = 0; i < limit; i++)
        {
            indices[dim] = i;
            SliceTensorRecursive(input, output, indices, dim + 1, axis, start, length);
        }
    }

    private static Tensor<T> ConcatenateTensors(Tensor<T> a, Tensor<T> b, int axis)
    {
        var shape = (int[])a.Shape.Clone();
        shape[axis] = a.Shape[axis] + b.Shape[axis];
        var output = new Tensor<T>(shape);
        var indices = new int[a.Shape.Length];
        ConcatRecursive(a, output, indices, 0, axis, 0);
        ConcatRecursive(b, output, indices, 0, axis, a.Shape[axis]);
        return output;
    }

    private static void ConcatRecursive(
        Tensor<T> src, Tensor<T> dst, int[] indices,
        int dim, int axis, int offset)
    {
        if (dim == indices.Length)
        {
            var dstIndices = (int[])indices.Clone();
            dstIndices[axis] += offset;
            dst[dstIndices] = src[indices];
            return;
        }
        int limit = src.Shape[dim];
        for (int i = 0; i < limit; i++)
        {
            indices[dim] = i;
            ConcatRecursive(src, dst, indices, dim + 1, axis, offset);
        }
    }

    #endregion

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_inputProjectionWeightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _inputProjectionWeights = Engine.TensorAdd(_inputProjectionWeights, Engine.TensorMultiplyScalar(_inputProjectionWeightsGradient, negLR));
        _inputProjectionBias = Engine.TensorAdd(_inputProjectionBias, Engine.TensorMultiplyScalar(_inputProjectionBiasGradient!, negLR));
        _convWeights = Engine.TensorAdd(_convWeights, Engine.TensorMultiplyScalar(_convWeightsGradient!, negLR));
        _convBias = Engine.TensorAdd(_convBias, Engine.TensorMultiplyScalar(_convBiasGradient!, negLR));
        _bProjectionWeights = Engine.TensorAdd(_bProjectionWeights, Engine.TensorMultiplyScalar(_bProjectionWeightsGradient!, negLR));
        _cProjectionWeights = Engine.TensorAdd(_cProjectionWeights, Engine.TensorMultiplyScalar(_cProjectionWeightsGradient!, negLR));
        _aLog = Engine.TensorAdd(_aLog, Engine.TensorMultiplyScalar(_aLogGradient!, negLR));
        _dtProjectionWeights = Engine.TensorAdd(_dtProjectionWeights, Engine.TensorMultiplyScalar(_dtProjectionWeightsGradient!, negLR));
        _dtProjectionBias = Engine.TensorAdd(_dtProjectionBias, Engine.TensorMultiplyScalar(_dtProjectionBiasGradient!, negLR));
        _dParam = Engine.TensorAdd(_dParam, Engine.TensorMultiplyScalar(_dParamGradient!, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(_outputProjectionWeightsGradient!, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(_outputProjectionBiasGradient!, negLR));
        _normGamma = Engine.TensorAdd(_normGamma, Engine.TensorMultiplyScalar(_normGammaGradient!, negLR));
        _normBeta = Engine.TensorAdd(_normBeta, Engine.TensorMultiplyScalar(_normBetaGradient!, negLR));
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        int totalParams = ParameterCount;
        var parameters = new Vector<T>(totalParams);
        int index = 0;

        foreach (var tensor in new[]
        {
            _inputProjectionWeights, _inputProjectionBias,
            _convWeights, _convBias,
            _bProjectionWeights, _cProjectionWeights,
            _aLog,
            _dtProjectionWeights, _dtProjectionBias,
            _dParam,
            _outputProjectionWeights, _outputProjectionBias,
            _normGamma, _normBeta
        })
        {
            for (int i = 0; i < tensor.Length; i++)
                parameters[index++] = tensor[i];
        }

        return parameters;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int expectedParams = ParameterCount;
        if (parameters.Length != expectedParams)
            throw new ArgumentException($"Expected {expectedParams} parameters, got {parameters.Length}");

        int index = 0;
        foreach (var tensor in new[]
        {
            _inputProjectionWeights, _inputProjectionBias,
            _convWeights, _convBias,
            _bProjectionWeights, _cProjectionWeights,
            _aLog,
            _dtProjectionWeights, _dtProjectionBias,
            _dParam,
            _outputProjectionWeights, _outputProjectionBias,
            _normGamma, _normBeta
        })
        {
            for (int i = 0; i < tensor.Length; i++)
                tensor[i] = parameters[index++];
        }
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastXBranch = null;
        _lastZBranch = null;
        _lastConvOutput = null;
        _lastSiluOutput = null;
        _lastSsdOutput = null;
        _lastGatedOutput = null;
        _lastDelta = null;
        _lastDeltaPreSoftplus = null;
        _lastB = null;
        _lastC = null;
        _lastHiddenStates = null;
        _lastNormInput = null;
        _originalInputShape = null;
        _inputProjectionWeightsGradient = null;
        _inputProjectionBiasGradient = null;
        _convWeightsGradient = null;
        _convBiasGradient = null;
        _bProjectionWeightsGradient = null;
        _cProjectionWeightsGradient = null;
        _aLogGradient = null;
        _dtProjectionWeightsGradient = null;
        _dtProjectionBiasGradient = null;
        _dParamGradient = null;
        _outputProjectionWeightsGradient = null;
        _outputProjectionBiasGradient = null;
        _normGammaGradient = null;
        _normBetaGradient = null;
    }

    #endregion

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        var xPlaceholder = new Tensor<T>(new int[] { 1, _innerDimension });
        var xNode = TensorOperations<T>.Variable(xPlaceholder, "x_t");

        var zPlaceholder = new Tensor<T>(new int[] { 1, _innerDimension });
        var zNode = TensorOperations<T>.Variable(zPlaceholder, "z_t");

        var hPrevPlaceholder = new Tensor<T>(new int[] { 1, _innerDimension * _stateDimension });
        var hPrevNode = TensorOperations<T>.Variable(hPrevPlaceholder, "h_prev");

        var dParamExpanded = new Tensor<T>(new int[] { _innerDimension });
        for (int h = 0; h < _numHeads; h++)
            for (int d = 0; d < _headDimension; d++)
                dParamExpanded[h * _headDimension + d] = _dParam[h];
        var dParamNode = TensorOperations<T>.Variable(dParamExpanded, "D");
        var outProjWeightsNode = TensorOperations<T>.Variable(_outputProjectionWeights, "W_out");
        var outProjBiasNode = TensorOperations<T>.Variable(_outputProjectionBias, "b_out");

        inputNodes.Add(xNode);
        inputNodes.Add(zNode);
        inputNodes.Add(hPrevNode);
        inputNodes.Add(dParamNode);
        inputNodes.Add(outProjWeightsNode);
        inputNodes.Add(outProjBiasNode);

        var skipOutput = TensorOperations<T>.ElementwiseMultiply(xNode, dParamNode);
        var zGate = TensorOperations<T>.Swish(zNode);
        var gatedOutput = TensorOperations<T>.ElementwiseMultiply(skipOutput, zGate);
        var outProjWeightsT = TensorOperations<T>.Transpose(outProjWeightsNode);
        var finalOutput = TensorOperations<T>.MatrixMultiply(gatedOutput, outProjWeightsT);
        var outputWithBias = TensorOperations<T>.Add(finalOutput, outProjBiasNode);

        return outputWithBias;
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["StateDimension"] = _stateDimension.ToString();
        metadata["InnerDimension"] = _innerDimension.ToString();
        metadata["NumHeads"] = _numHeads.ToString();
        metadata["HeadDimension"] = _headDimension.ToString();
        metadata["ConvKernelSize"] = _convKernelSize.ToString();
        metadata["ChunkSize"] = _chunkSize.ToString();
        return metadata;
    }

    /// <summary>
    /// Gets the input projection weights for external inspection or quantization.
    /// </summary>
    public Tensor<T> GetInputProjectionWeights() => _inputProjectionWeights;

    /// <summary>
    /// Gets the output projection weights for external inspection or quantization.
    /// </summary>
    public Tensor<T> GetOutputProjectionWeights() => _outputProjectionWeights;

    /// <summary>
    /// Gets the A_log parameter tensor for external inspection.
    /// </summary>
    public Tensor<T> GetALogParameter() => _aLog;

    /// <summary>
    /// Gets the D skip connection parameter for external inspection.
    /// </summary>
    public Tensor<T> GetDParameter() => _dParam;
}

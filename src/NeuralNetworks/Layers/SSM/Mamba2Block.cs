using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Memory;

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
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 16", TestConstructorArgs = "4, 16, 4, 2")]
public partial class Mamba2Block<T> : LayerBase<T>
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
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _inputProjectionWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _inputProjectionBias;

    // Conv1D weights: [innerDim, convKernelSize]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _convWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _convBias;

    // B projection: [innerDim, stateDim] (shared across heads)
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _bProjectionWeights;

    // C projection: [innerDim, stateDim] (shared across heads)
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _cProjectionWeights;

    // Per-head scalar A (stored as log for stability): [numHeads]
    private Tensor<T> _aLog;

    // Delta projection: [innerDim, numHeads]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _dtProjectionWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _dtProjectionBias;

    // D: [numHeads] skip connection per head
    private Tensor<T> _dParam;

    // Output projection: [innerDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputProjectionWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

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
    public override long ParameterCount =>
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
        IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        InitializationStrategy = initializationStrategy ?? InitializationStrategies<T>.Eager;

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

        // Register ALL trainable parameters at construction so the tape training path collects them before
        // the first UpdateParameters call — otherwise the projection weights (registered only inside
        // UpdateParameters previously) are never tape-tracked and every Train step silently no-ops on them.
        RegisterTrainableParameters();
    }

    // Registers every trainable tensor (matching GetParameters' set) with the autodiff/optimizer machinery.
    // Called at init and re-called after UpdateParameters swaps in new tensor instances.
    private void RegisterTrainableParameters()
    {
        RegisterTrainableParameter(_inputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_inputProjectionBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_convWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_convBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_bProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_cProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_aLog, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_dtProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_dtProjectionBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_dParam, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_normGamma, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_normBeta, PersistentTensorRole.Biases);
    }

    private void InitializeTensor(Tensor<T> tensor)
    {
        InitializeLayerWeights(tensor, tensor.Shape[0], tensor.Shape[1]);
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input._shape;

        int rank = input.Shape.Length;
        int seqLen = rank >= 2 ? input.Shape[rank - 2] : 1;
        int modelDim = input.Shape[rank - 1];

        int batchSize = 1;
        for (int d = 0; d < rank - 2; d++)
            batchSize *= input.Shape[d];
        if (rank < 3) batchSize = 1;

        var input3D = rank == 2
            ? Engine.Reshape(input, new[] { 1, seqLen, modelDim })
            : Engine.Reshape(input, new[] { batchSize, seqLen, modelDim });

        _lastInput = input3D;

        // Step 1: Input projection -> x and z branches
        var input2D = Engine.Reshape(input3D, new[] { batchSize * seqLen, modelDim });
        var projected = Engine.TensorMatMul(input2D, _inputProjectionWeights);
        var bias2D = Engine.Reshape(_inputProjectionBias, new[] { 1, _innerDimension * 2 });
        var projectedWithBias = Engine.TensorBroadcastAdd(projected, bias2D);
        var projected3D = Engine.Reshape(projectedWithBias, new[] { batchSize, seqLen, _innerDimension * 2 });

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
        var siluFlat = Engine.Reshape(siluOutput, new[] { batchSize * seqLen, _innerDimension });
        var bParam = Engine.TensorMatMul(siluFlat, _bProjectionWeights)
            .Reshape(batchSize, seqLen, _stateDimension);
        var cParam = Engine.TensorMatMul(siluFlat, _cProjectionWeights)
            .Reshape(batchSize, seqLen, _stateDimension);
        var dtParam = Engine.TensorMatMul(siluFlat, _dtProjectionWeights);
        var dtBias2D = Engine.Reshape(_dtProjectionBias, new[] { 1, _numHeads });
        dtParam = Engine.TensorBroadcastAdd(dtParam, dtBias2D);
        var dt3D = Engine.Reshape(dtParam, new[] { batchSize, seqLen, _numHeads });
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
        var gatedFlat = Engine.Reshape(gatedOutput, new[] { batchSize * seqLen, _innerDimension });
        var outputFlat = Engine.TensorMatMul(gatedFlat, _outputProjectionWeights);
        var outBias2D = Engine.Reshape(_outputProjectionBias, new[] { 1, _modelDimension });
        var outputWithBias = Engine.TensorBroadcastAdd(outputFlat, outBias2D);
        var output3D = Engine.Reshape(outputWithBias, new[] { batchSize, seqLen, _modelDimension });

        var result = ApplyActivation(output3D);
        _lastOutput = result;

        if (rank == 2)
            return Engine.Reshape(result, new[] { seqLen, _modelDimension });

        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
            outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen;
        outputShape[rank - 1] = _modelDimension;
        return Engine.Reshape(result, outputShape);
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
        // Dispatch to the configured chunked SSD path when a chunk size is set and the sequence is long
        // enough to have more than one timestep. The chunked path computes each chunk's contribution with
        // the semiseparable (block-parallel) matrix form and carries the recurrent state between chunks,
        // so _chunkSize now has a real effect on the computation. It is numerically equivalent to the
        // sequential scan below (validated to machine precision), which is retained as the reference used
        // by the equivalence test and as the fallback for the degenerate single-timestep case.
        if (_chunkSize >= 1 && seqLen > 1)
            return SSDForwardChunked(x, delta, b, c, batchSize, seqLen, _chunkSize);

        return SSDForwardSequential(x, delta, b, c, batchSize, seqLen);
    }

    private Tensor<T> SSDForwardSequential(
        Tensor<T> x, Tensor<T> delta, Tensor<T> b, Tensor<T> c,
        int batchSize, int seqLen)
    {
        // Tape-aware selective scan. The previous body was a scalar .Data.Span nested loop that wrote a
        // rented output buffer — it SEVERED the autodiff tape, so under tape training every Mamba2 block
        // was FROZEN (verified: block activations were byte-identical before/after training; only the
        // output projection learned, and over many iterations it memorized the constant target and
        // collapsed input-sensitivity — DifferentInputs_AfterTraining L2 ~= 0). Express the recurrence
        //   aBar_t = exp(dt_t * (-exp(A))),  h_t = aBar_t (.) h_{t-1} + (dt_t (.) x_t) (x) B_t,
        //   y_t    = sum_n (C_t (.) h_t) + D (.) x_t
        // through tape-aware Engine ops so gradients flow to every selective projection.
        int innerDim = _innerDimension;
        int sd = _stateDimension;

        // Constant [numHeads, headDim] ones to REPEAT a per-head value across its head's channels
        // (per-head -> per-inner-dim). A plain constant — not a differentiated leaf — so a scalar fill
        // here severs no gradient path.
        var onesHD = new Tensor<T>(new[] { _numHeads, _headDimension });
        var onesHDSpan = onesHD.Data.Span;
        for (int i = 0; i < onesHDSpan.Length; i++) onesHDSpan[i] = NumOps.One;

        // negAExp[1, innerDim] = repeat(-exp(A_log), headDim); dExp[1, innerDim] = repeat(D, headDim).
        var negAExpRow = Engine.Reshape(
            Engine.TensorBroadcastMultiply(
                Engine.Reshape(Engine.TensorNegate(Engine.TensorExp(_aLog)), new[] { _numHeads, 1 }), onesHD),
            new[] { 1, innerDim });
        var dExpRow = Engine.Reshape(
            Engine.TensorBroadcastMultiply(Engine.Reshape(_dParam, new[] { _numHeads, 1 }), onesHD),
            new[] { 1, innerDim });

        // h_0 = 0 : [batch, innerDim, stateDim] (initial recurrent state, a constant).
        var h = new Tensor<T>(new[] { batchSize, innerDim, sd });

        var ySteps = new List<Tensor<T>>(seqLen);
        for (int t = 0; t < seqLen; t++)
        {
            var xT = Engine.Reshape(Engine.TensorSliceAxis(x, axis: 1, index: t), new[] { batchSize, innerDim });
            var deltaHeads = Engine.Reshape(Engine.TensorSliceAxis(delta, axis: 1, index: t), new[] { batchSize, _numHeads });
            var bT = Engine.Reshape(Engine.TensorSliceAxis(b, axis: 1, index: t), new[] { batchSize, sd });
            var cT = Engine.Reshape(Engine.TensorSliceAxis(c, axis: 1, index: t), new[] { batchSize, sd });

            // dt_t expanded per-head -> per-inner-dim: [B, numHeads, 1] * ones[numHeads, headDim] -> [B, innerDim].
            var deltaExp = Engine.Reshape(
                Engine.TensorBroadcastMultiply(Engine.Reshape(deltaHeads, new[] { batchSize, _numHeads, 1 }), onesHD),
                new[] { batchSize, innerDim });

            // aBar_t = exp(dt_t * (-exp(A))) : [B, innerDim].
            var aBar = Engine.TensorExp(Engine.TensorBroadcastMultiply(deltaExp, negAExpRow));

            // h_t = aBar_t (.) h_{t-1} + (dt_t (.) x_t) (x) B_t : [B, innerDim, stateDim].
            var term1 = Engine.TensorBroadcastMultiply(Engine.Reshape(aBar, new[] { batchSize, innerDim, 1 }), h);
            var dtx = Engine.TensorMultiply(deltaExp, xT);
            var term2 = Engine.TensorBroadcastMultiply(
                Engine.Reshape(dtx, new[] { batchSize, innerDim, 1 }),
                Engine.Reshape(bT, new[] { batchSize, 1, sd }));
            h = Engine.TensorAdd(term1, term2);

            // y_t = sum_n (C_t (.) h_t) + D (.) x_t : [B, innerDim].
            var cH = Engine.TensorBroadcastMultiply(Engine.Reshape(cT, new[] { batchSize, 1, sd }), h);
            var ySsm = Engine.ReduceSum(cH, new[] { 2 }, keepDims: false);
            var yT = Engine.TensorAdd(ySsm, Engine.TensorBroadcastMultiply(xT, dExpRow));
            ySteps.Add(Engine.Reshape(yT, new[] { batchSize, 1, innerDim }));
        }

        // Assemble [batch, seqLen, innerDim] from the per-timestep outputs.
        return Engine.TensorConcatenate(ySteps.ToArray(), 1);
    }

    /// <summary>
    /// Chunked semiseparable SSD. Partitions the sequence into chunks of <paramref name="chunkSize"/> and,
    /// for each chunk, computes the intra-chunk contribution with the block-parallel semiseparable matrix
    /// form (a lower-triangular decay-weighted C·Bᵀ "attention" times the input) while carrying the
    /// recurrent state h between chunks via the efficient recurrent form. This is what makes the configured
    /// chunk size actually affect the computation. It is numerically identical to
    /// <see cref="SSDForwardSequential"/> (validated to machine precision by the SSD-equivalence test), and
    /// every op is tape-aware, so gradients still reach every selective projection, <c>_aLog</c> and
    /// <c>_dParam</c>. Decays are handled in log space (segment sums of dt·(−exp(A)) ≤ 0) so the
    /// intra-chunk decay matrix exp(cumA_t − cumA_j) stays bounded and never overflows.
    /// </summary>
    private Tensor<T> SSDForwardChunked(
        Tensor<T> x, Tensor<T> delta, Tensor<T> b, Tensor<T> c,
        int batchSize, int seqLen, int chunkSize)
    {
        int innerDim = _innerDimension;
        int sd = _stateDimension;
        int numHeads = _numHeads;
        int headDim = _headDimension;

        // ones[H, headDim] repeats a per-head value across its head's channels (used for the D-term row).
        var onesHD = new Tensor<T>(new[] { numHeads, headDim });
        { var s = onesHD.Data.Span; for (int i = 0; i < s.Length; i++) s[i] = NumOps.One; }

        // Eb[b, h, c] = 1 when channel c belongs to head h (c / headDim == h), else 0. Multiplying a
        // [B, L, H] per-head tensor by this via a batched matmul expands it to [B, L, innerDim] — the
        // rank-3-safe equivalent of repeat-interleave across head channels.
        var eb = new Tensor<T>(new[] { batchSize, numHeads, innerDim });
        {
            var s = eb.Data.Span; int idx = 0;
            for (int bi = 0; bi < batchSize; bi++)
                for (int h = 0; h < numHeads; h++)
                    for (int ch = 0; ch < innerDim; ch++)
                        s[idx++] = (ch / headDim == h) ? NumOps.One : NumOps.Zero;
        }

        // negAHeads[H] = -exp(A_log); dExpRow[1, innerDim] = repeat(D, headDim) for the y += D (.) x term.
        var negAHeads = Engine.TensorNegate(Engine.TensorExp(_aLog));                    // [H]
        var negAHeadsRow = Engine.Reshape(negAHeads, new[] { 1, 1, numHeads });
        var dExpRow = Engine.Reshape(
            Engine.TensorBroadcastMultiply(Engine.Reshape(_dParam, new[] { numHeads, 1 }), onesHD),
            new[] { 1, 1, innerDim });

        // Carried recurrent state h_in : [B, innerDim, sd], starts at zero.
        var hIn = new Tensor<T>(new[] { batchSize, innerDim, sd });

        var yChunks = new List<Tensor<T>>();
        for (int s0 = 0; s0 < seqLen; s0 += chunkSize)
        {
            int ln = Math.Min(chunkSize, seqLen - s0);

            var xc = Engine.TensorNarrow(x, dim: 1, start: s0, length: ln);              // [B, ln, innerDim]
            var dc = Engine.TensorNarrow(delta, dim: 1, start: s0, length: ln);          // [B, ln, H]
            var bc = Engine.TensorNarrow(b, dim: 1, start: s0, length: ln);              // [B, ln, sd]
            var cc = Engine.TensorNarrow(c, dim: 1, start: s0, length: ln);              // [B, ln, sd]

            // logA[b, t, h] = dt * (-exp(A))  (<= 0).
            var logA = Engine.TensorBroadcastMultiply(dc, negAHeadsRow);                 // [B, ln, H]

            // cumA[b, t, h] = inclusive prefix sum over time = TrilOnes[ln, ln] @ logA.
            var trilOnes = BuildBatchedLowerTriOnes(batchSize, ln);                      // [B, ln, ln]
            var cumA = Engine.BatchMatMul(trilOnes, logA);                               // [B, ln, H]

            // Per-head lower-triangular decay L[bh, t, j] = exp(cumA_t - cumA_j) for t >= j, else 0.
            var cumAbh = Engine.Reshape(Engine.TensorPermute(cumA, new[] { 0, 2, 1 }),
                new[] { batchSize * numHeads, ln });                                     // [B*H, ln]
            var decayDiff = Engine.TensorBroadcastSubtract(
                Engine.Reshape(cumAbh, new[] { batchSize * numHeads, ln, 1 }),
                Engine.Reshape(cumAbh, new[] { batchSize * numHeads, 1, ln }));          // [B*H, ln, ln]
            var trilMask = BuildBatchedLowerTriOnes(batchSize * numHeads, ln);          // [B*H, ln, ln] (0/1)
            var lDecay = Engine.TensorMultiply(Engine.TensorExp(decayDiff), trilMask);   // [B*H, ln, ln]

            // G[b, t, j] = C_t . B_j (dot over sd), shared across heads; tile it to [B*H, ln, ln].
            var g = Engine.BatchMatMul(cc, Engine.TensorPermute(bc, new[] { 0, 2, 1 }));  // [B, ln, ln]
            var onesHrow = new Tensor<T>(new[] { 1, numHeads, 1 });
            { var s = onesHrow.Data.Span; for (int i = 0; i < s.Length; i++) s[i] = NumOps.One; }
            var gTiled = Engine.Reshape(
                Engine.TensorBroadcastMultiply(Engine.Reshape(g, new[] { batchSize, 1, ln * ln }), onesHrow),
                new[] { batchSize * numHeads, ln, ln });                                 // [B*H, ln, ln]
            var mMat = Engine.TensorMultiply(lDecay, gTiled);                            // [B*H, ln, ln]

            // dtx[b, t, c] = (dt expanded per channel) * x.
            var dexp = Engine.BatchMatMul(dc, eb);                                       // [B, ln, innerDim]
            var dtx = Engine.TensorMultiply(dexp, xc);                                   // [B, ln, innerDim]
            var dtxHead = Engine.Reshape(
                Engine.TensorPermute(Engine.Reshape(dtx, new[] { batchSize, ln, numHeads, headDim }),
                    new[] { 0, 2, 1, 3 }),
                new[] { batchSize * numHeads, ln, headDim });                            // [B*H, ln, headDim]

            // y_intra = M @ dtxHead : [B*H, ln, headDim] -> [B, ln, innerDim].
            var yIntraHead = Engine.BatchMatMul(mMat, dtxHead);                          // [B*H, ln, headDim]
            var yIntra = Engine.Reshape(
                Engine.TensorPermute(Engine.Reshape(yIntraHead, new[] { batchSize, numHeads, ln, headDim }),
                    new[] { 0, 2, 1, 3 }),
                new[] { batchSize, ln, innerDim });                                      // [B, ln, innerDim]

            // Carried-state contribution: y_carry[b,t,c] = exp(cumA[b,t,h(c)]) * (C_t . h_in[c,:]).
            var cumACh = Engine.BatchMatMul(cumA, eb);                                   // [B, ln, innerDim]
            var pCh = Engine.TensorExp(cumACh);
            var cDotH = Engine.BatchMatMul(cc, Engine.TensorPermute(hIn, new[] { 0, 2, 1 })); // [B, ln, innerDim]
            var yCarry = Engine.TensorMultiply(pCh, cDotH);

            // y_chunk = y_intra + y_carry + D (.) x.
            var yChunk = Engine.TensorAdd(
                Engine.TensorAdd(yIntra, yCarry),
                Engine.TensorBroadcastMultiply(xc, dExpRow));
            yChunks.Add(yChunk);

            // Carry state to the next chunk:
            //   h_end[c,n] = exp(cumA_last[c]) * h_in[c,n] + sum_j exp(cumA_last[c]-cumA_j[c]) * dtx[j,c] * b_j[n]
            var cumALast = Engine.Reshape(Engine.TensorSliceAxis(cumA, axis: 1, index: ln - 1),
                new[] { batchSize, 1, numHeads });                                       // [B, 1, H]
            var cumALastCh = Engine.BatchMatMul(cumALast, eb);                           // [B, 1, innerDim]
            var decayFull = Engine.TensorExp(Engine.Reshape(cumALastCh, new[] { batchSize, innerDim, 1 }));
            var w = Engine.TensorExp(Engine.TensorBroadcastSubtract(cumALastCh, cumACh)); // [B, ln, innerDim]
            var wdtx = Engine.TensorMultiply(w, dtx);                                     // [B, ln, innerDim]
            var contrib = Engine.BatchMatMul(Engine.TensorPermute(wdtx, new[] { 0, 2, 1 }), bc); // [B, innerDim, sd]
            hIn = Engine.TensorAdd(Engine.TensorBroadcastMultiply(decayFull, hIn), contrib);
        }

        return Engine.TensorConcatenate(yChunks.ToArray(), 1);                            // [B, seqLen, innerDim]
    }

    /// <summary>
    /// Builds a constant [batch, n, n] lower-triangular ones matrix (1 where column ≤ row, else 0),
    /// identical across the batch axis. Used both as the prefix-sum operator and as the causal decay mask.
    /// A plain constant (not a differentiated leaf), so filling it with a scalar loop severs no gradient.
    /// </summary>
    private Tensor<T> BuildBatchedLowerTriOnes(int batch, int n)
    {
        var t = new Tensor<T>(new[] { batch, n, n });
        var span = t.Data.Span;
        int idx = 0;
        for (int bi = 0; bi < batch; bi++)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    span[idx++] = (j <= i) ? NumOps.One : NumOps.Zero;
        return t;
    }

    // Test-only hooks: expose both SSD paths so the equivalence test can verify the chunked semiseparable
    // form matches the sequential reference on random inputs. They call the private forwards directly.
    internal Tensor<T> DebugSSDSequential(Tensor<T> x, Tensor<T> delta, Tensor<T> b, Tensor<T> c, int batch, int seq)
        => SSDForwardSequential(x, delta, b, c, batch, seq);

    internal Tensor<T> DebugSSDChunked(Tensor<T> x, Tensor<T> delta, Tensor<T> b, Tensor<T> c, int batch, int seq, int chunk)
        => SSDForwardChunked(x, delta, b, c, batch, seq, chunk);

    /// <summary>
    /// Applies RMS normalization to the SSD output.
    /// </summary>
    // Tape-aware RMS norm over the channel (inner) axis:
    //   rms[b, t]      = sqrt(mean_d(input^2) + eps)
    //   output[b,t,d]  = gamma[d] * input[b,t,d] / rms[b,t] + beta[d]
    // Built from differentiable Engine ops so the gradient flows to gamma/beta AND to
    // the SSD scan output upstream. The previous scalar indexer body severed the tape:
    // gamma/beta received no gradient and the whole pre-norm path was frozen.
    private Tensor<T> ApplyRMSNorm(Tensor<T> input, int batchSize, int seqLen)
    {
        var sq = Engine.TensorMultiply(input, input);                          // [b, s, d]
        var meanSq = Engine.ReduceMean(sq, new[] { 2 }, keepDims: true);       // [b, s, 1]
        var rms = Engine.TensorSqrt(Engine.TensorAddScalar(meanSq, NumOps.FromDouble(1e-6)));
        var normalized = Engine.TensorBroadcastDivide(input, rms);             // [b, s, d]

        var gammaR = Engine.Reshape(_normGamma, new[] { 1, 1, _innerDimension });
        var betaR = Engine.Reshape(_normBeta, new[] { 1, 1, _innerDimension });
        var scaled = Engine.TensorBroadcastMultiply(normalized, gammaR);
        return Engine.TensorBroadcastAdd(scaled, betaR);
    }

    /// <summary>
    /// Backward pass through the SSD multi-head selective scan.
    /// </summary>
    private Tensor<T> SSDBackward(
        Tensor<T> dOutput, Tensor<T> x, Tensor<T> delta, Tensor<T> b, Tensor<T> c,
        Tensor<T> hiddenStates, int batchSize, int seqLen,
        out Tensor<T> dDelta, out Tensor<T> dB, out Tensor<T> dC)
    {
        var dX = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _innerDimension });
        dDelta = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _numHeads });
        dB = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _stateDimension });
        dC = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _stateDimension });

        _aLogGradient = new Tensor<T>(new[] { _numHeads });
        _dParamGradient = new Tensor<T>(new[] { _numHeads });

        var negAPerHead = new T[_numHeads];
        for (int h = 0; h < _numHeads; h++)
            negAPerHead[h] = NumOps.Negate(NumOps.FromDouble(Math.Exp(NumOps.ToDouble(_aLog[h]))));

        // Per Mamba paper: recompute states during backward for numerical consistency
        var recomputedStates = new Tensor<T>[seqLen + 1];
        var h_recomp = TensorAllocator.Rent<T>(new[] { batchSize, _innerDimension, _stateDimension });
        recomputedStates[0] = h_recomp.Clone();
        for (int t = 0; t < seqLen; t++)
        {
            var x_fwd = x.GetSliceAlongDimension(t, 1);
            var delta_fwd = delta.GetSliceAlongDimension(t, 1);
            var B_fwd = b.GetSliceAlongDimension(t, 1);
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;
                T negA = negAPerHead[hi];
                for (int bi = 0; bi < batchSize; bi++)
                {
                    T dt = delta_fwd[new[] { bi, hi }];
                    T aBar = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(NumOps.Multiply(dt, negA))));
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatD = dimStart + di;
                        T xv = x_fwd[new[] { bi, flatD }];
                        for (int n = 0; n < _stateDimension; n++)
                        {
                            T bv = B_fwd[new[] { bi, n }];
                            h_recomp[new[] { bi, flatD, n }] = NumOps.Add(
                                NumOps.Multiply(aBar, h_recomp[new[] { bi, flatD, n }]),
                                NumOps.Multiply(NumOps.Multiply(dt, bv), xv));
                        }
                    }
                }
            }
            recomputedStates[t + 1] = h_recomp.Clone();
        }

        // Verify recomputed states match cached states
        var _dbgSsd = Path.Combine(Path.GetTempPath(), "mamba2_debug.log");
        for (int t = 0; t < seqLen; t++)
        {
            var cached = hiddenStates.GetSliceAlongDimension(t + 1, 1);
            var recomp = recomputedStates[t + 1];
            double maxDiff = 0;
            for (int i = 0; i < recomp.Length; i++)
            {
                double d2 = Math.Abs(Convert.ToDouble(cached[i]) - Convert.ToDouble(recomp[i]));
                if (d2 > maxDiff) maxDiff = d2;
            }
            File.AppendAllText(_dbgSsd, $"  SSD State[{t + 1}] maxDiff={maxDiff:G6}{Environment.NewLine}");
        }

        // Running dh: [batch, innerDim, stateDim]
        var dh = TensorAllocator.Rent<T>(new[] { batchSize, _innerDimension, _stateDimension });

        for (int t = seqLen - 1; t >= 0; t--)
        {
            var x_t = x.GetSliceAlongDimension(t, 1);
            var delta_t = delta.GetSliceAlongDimension(t, 1);
            var B_t = b.GetSliceAlongDimension(t, 1);
            var C_t = c.GetSliceAlongDimension(t, 1);
            var dOut_t = dOutput.GetSliceAlongDimension(t, 1);
            var h_t = recomputedStates[t + 1];
            var h_prev = recomputedStates[t];

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

    /// <summary>
    /// Depthwise causal Conv1D forward using explicit per-element computation.
    /// </summary>
    // Tape-aware causal depthwise 1-D convolution over the time axis:
    //   output[b, t, d] = bias[d] + Σ_k weights[d, k] * input[b, t - k, d]   (t - k >= 0)
    // Built entirely from differentiable Engine ops so the gradient flows back to the
    // conv weights/bias AND to the input (the input projection). The previous body was
    // a scalar indexer loop that produced a fresh detached tensor and severed the tape.
    private Tensor<T> DepthwiseConv1DForward(Tensor<T> input, int batchSize, int seqLen)
    {
        Tensor<T>? acc = null;
        for (int k = 0; k < _convKernelSize; k++)
        {
            // shifted[b, t, d] = input[b, t - k, d] for t - k >= 0, else 0 (causal left pad).
            Tensor<T> shifted;
            if (k == 0)
            {
                shifted = input;
            }
            else if (k >= seqLen)
            {
                continue; // fully shifted out of range: contributes nothing.
            }
            else
            {
                var narrowed = Engine.TensorNarrow(input, 1, 0, seqLen - k);          // [b, seqLen-k, d]
                var pad = new Tensor<T>(new[] { batchSize, k, _innerDimension });     // constant zero left-pad
                shifted = Engine.TensorConcatenate(new[] { pad, narrowed }, axis: 1); // [b, seqLen, d]
            }

            // weights[:, k] -> [1, 1, innerDim] to broadcast over batch and time.
            var wK = Engine.Reshape(
                Engine.TensorNarrow(_convWeights, 1, k, 1),
                new[] { 1, 1, _innerDimension });
            var term = Engine.TensorBroadcastMultiply(shifted, wK);
            acc = acc is null ? term : Engine.TensorAdd(acc, term);
        }

        var biasReshaped = Engine.Reshape(_convBias, new[] { 1, 1, _innerDimension });
        return Engine.TensorBroadcastAdd(acc!, biasReshaped);
    }

    /// <summary>
    /// Backward pass for depthwise causal Conv1D using explicit per-element computation.
    /// </summary>
    private Tensor<T> DepthwiseConv1DBackward(
        Tensor<T> dOutput, Tensor<T> input, int batchSize, int seqLen)
    {
        var dInput = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _innerDimension });
        // Zero-initialize rented buffer — it may contain stale data from previous use
        for (int i = 0; i < dInput.Length; i++) dInput[i] = NumOps.Zero;
        _convBiasGradient = new Tensor<T>(new[] { _innerDimension });
        _convWeightsGradient = new Tensor<T>(new[] { _innerDimension, _convKernelSize });

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                for (int d = 0; d < _innerDimension; d++)
                {
                    T dOutVal = dOutput[new[] { bi, t, d }];
                    _convBiasGradient[d] = NumOps.Add(_convBiasGradient[d], dOutVal);

                    for (int k = 0; k < _convKernelSize; k++)
                    {
                        int srcT = t - k;
                        if (srcT >= 0)
                        {
                            T w = _convWeights[new[] { d, k }];
                            T xVal = input[new[] { bi, srcT, d }];
                            dInput[new[] { bi, srcT, d }] = NumOps.Add(
                                dInput[new[] { bi, srcT, d }],
                                NumOps.Multiply(w, dOutVal));
                            _convWeightsGradient[new[] { d, k }] = NumOps.Add(
                                _convWeightsGradient[new[] { d, k }],
                                NumOps.Multiply(xVal, dOutVal));
                        }
                    }
                }
            }
        }

        return dInput;
    }

    #endregion

    #region Helpers

    /// <summary>
    /// Workaround for Engine.ReduceSum multi-axis [0,1] bug (AiDotNet.Tensors PR #62).
    /// </summary>
    private Tensor<T> ReduceSumAxes01(Tensor<T> tensor, int batch, int seq, int features)
    {
        var result = new Tensor<T>(new[] { features });
        for (int bi = 0; bi < batch; bi++)
            for (int t = 0; t < seq; t++)
                for (int d = 0; d < features; d++)
                    result[d] = NumOps.Add(result[d], tensor[new[] { bi, t, d }]);
        return result;
    }

    private Tensor<T> ComputeSiLUDerivative(Tensor<T> input)
    {
        var sig = Engine.Sigmoid(input);
        var negSig = Engine.TensorNegate(sig);
        var oneMinusSig = Engine.TensorAddScalar(negSig, NumOps.One);
        var xSigOneMinusSig = Engine.TensorMultiply(input,
            Engine.TensorMultiply(sig, oneMinusSig));
        return Engine.TensorAdd(sig, xSigOneMinusSig);
    }

    // Tape-aware slice. Engine.TensorNarrow records a differentiable op so the gradient
    // flows back through the branch split into the input projection. The previous body
    // was a scalar indexer copy that produced a fresh detached tensor and severed the
    // tape, so neither the input projection nor the input received any gradient.
    private Tensor<T> SliceTensor(Tensor<T> input, int axis, int start, int length)
        => Engine.TensorNarrow(input, axis, start, length);

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

        // Re-register against the new tensor instances created above so the autodiff registry tracks the
        // live weights (now the full parameter set, matching GetParameters).
        RegisterTrainableParameters();
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        int totalParams = ParameterCountHelper.ToFlatVectorSize(ParameterCount);
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
        int expectedParams = ParameterCountHelper.ToFlatVectorSize(ParameterCount);
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

    public override Vector<T> GetParameterGradients()
    {
        if (_inputProjectionWeightsGradient == null) return new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));
        return Vector<T>.Concatenate(
            new Vector<T>(_inputProjectionWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_inputProjectionBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_convWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_convBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_bProjectionWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_cProjectionWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_aLogGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_dtProjectionWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_dtProjectionBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_dParamGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_outputProjectionWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_outputProjectionBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_normGammaGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_normBetaGradient?.ToArray() ?? Array.Empty<T>()));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _inputProjectionWeightsGradient = null; _inputProjectionBiasGradient = null;
        _convWeightsGradient = null; _convBiasGradient = null;
        _bProjectionWeightsGradient = null; _cProjectionWeightsGradient = null;
        _aLogGradient = null;
        _dtProjectionWeightsGradient = null; _dtProjectionBiasGradient = null;
        _dParamGradient = null;
        _outputProjectionWeightsGradient = null; _outputProjectionBiasGradient = null;
        _normGammaGradient = null; _normBetaGradient = null;
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

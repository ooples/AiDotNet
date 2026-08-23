using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements a general-purpose Mamba block (Selective State Space Model) from Gu and Dao, 2023.
/// </summary>
/// <remarks>
/// <para>
/// The Mamba block is the core building block of the Mamba architecture, which processes sequences
/// with O(n) linear time complexity compared to O(n^2) for standard Transformer attention.
/// It uses a selective scan mechanism (S6) where the state space parameters (A, B, C, delta)
/// are input-dependent, allowing the model to selectively propagate or forget information
/// along the sequence dimension.
/// </para>
/// <para>
/// The block follows the architecture: input projection -> Conv1D -> SiLU -> selective scan -> output gating -> output projection.
/// The selective scan implements the core SSM recurrence:
/// <code>
///   h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
///   y_t = C_t * h_t
/// </code>
/// where A_bar and B_bar are the Zero-Order Hold (ZOH) discretized versions of continuous parameters.
/// </para>
/// <para>
/// All within-timestep computations use hardware-accelerated Engine tensor operations (SIMD/AVX/GPU),
/// with only the sequential time loop remaining as scalar iteration. This enables full CPU/GPU
/// acceleration for the dominant O(batch * innerDim * stateDim) work per timestep.
/// </para>
/// <para><b>For Beginners:</b> Mamba is a modern alternative to Transformer attention that is much
/// faster for long sequences.
///
/// Think of how attention works in a Transformer:
/// - Every token looks at every other token -> O(n^2) cost
/// - For 1000 tokens, that's 1,000,000 comparisons
///
/// Mamba works differently:
/// - It maintains a "hidden state" that summarizes what it has seen so far
/// - Each new token updates this state and produces an output
/// - The key innovation is that HOW the state is updated depends on the input
/// - This "selective" mechanism lets it remember important tokens and forget irrelevant ones
///
/// The result: O(n) linear cost with performance competitive with Transformers.
/// Used by Falcon Mamba 7B, Jamba (AI21), Zamba (Zyphra), and many research models.
/// </para>
/// <para>
/// <b>Reference:</b> Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2024.
/// https://arxiv.org/abs/2312.00752
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 16", TestConstructorArgs = "4, 16, 4")]
// Shape-preserving, and the residual connection is what makes that a hard guarantee rather than a
// coincidence: ForwardTraced ends with Engine.TensorAdd(output3D, input3D), which only type-checks when
// the input's trailing width already equals _modelDimension. So every axis is carried through and
// nothing needs a hand-written relation - the generator derives Same(role) for each.
//
// Rank 2 is [Time, Features], not [Batch, Features]: ForwardTraced reads the sequence length from
// input.Shape[rank - 2] at every rank, and the constructor's base shape is [-1, modelDimension] with the
// -1 documented as the FREE sequence axis. That also matches [LayerProperty(TestInputShape = "4, 16")].
// Both ranks are spelled out rather than folded into one BatchOptional declaration because the generator
// emits one arm per declared axis count, and the optional-batch form would leave rank 2 without one.
//
// Rank 1 is deliberately absent: the output-reshape block indexes outputShape[rank - 2], so a rank-1
// input throws rather than round-tripping.
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output,
    Note = "Selective scan is a recurrence over Time: the sequence length is consumed, never resized.")]
[AutoParameters]
public partial class MambaBlock<T> : LayerBase<T>, IShapeContract
{
    // Configuration
    private readonly int _modelDimension;
    private readonly int _stateDimension;
    private readonly int _innerDimension;
    private readonly int _convKernelSize;
    private readonly int _dtRank;

    // Input projection: [modelDim, innerDim * 2] (projects to x and z branches)
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _inputProjectionWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _inputProjectionBias;

    // Conv1D weights: [innerDim, convKernelSize] (depthwise over sequence)
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _convWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _convBias;

    // SSM parameter projections (from inner dimension after Conv1D + SiLU)
    // x_proj: [innerDim, dtRank + stateDim * 2] (projects to delta, B, C)
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _xProjectionWeights;

    // dt_proj: [dtRank, innerDim] (projects dt from low rank to inner dimension)
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _dtProjectionWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _dtProjectionBias;

    // SSM continuous parameters
    // A: [innerDim, stateDim] (structured as -exp(A_log) for stability)
    private Tensor<T> _aLog;
    // D: [innerDim] (skip connection parameter)
    private Tensor<T> _dParam;

    // Output projection: [innerDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputProjectionWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _outputProjectionBias;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastXBranch;
    private Tensor<T>? _lastZBranch;
    private Tensor<T>? _lastConvOutput;
    private Tensor<T>? _lastSiluOutput;
    private Tensor<T>? _lastScanOutput;
    private Tensor<T>? _lastGatedOutput;
    private Tensor<T>? _lastDelta;
    private Tensor<T>? _lastDeltaPreSoftplus;
    private Tensor<T>? _lastB;
    private Tensor<T>? _lastC;
    private Tensor<T>? _lastHiddenStates;
    private Tensor<T>? _initialHiddenState;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _inputProjectionWeightsGradient;
    private Tensor<T>? _inputProjectionBiasGradient;
    private Tensor<T>? _convWeightsGradient;
    private Tensor<T>? _convBiasGradient;
    private Tensor<T>? _xProjectionWeightsGradient;
    private Tensor<T>? _dtProjectionWeightsGradient;
    private Tensor<T>? _dtProjectionBiasGradient;
    private Tensor<T>? _aLogGradient;
    private Tensor<T>? _dParamGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Every weight in this block is sized from constructor arguments, so the parameter surface is
    /// fully known before the first forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <c>modelDimension</c>, <c>stateDimension</c>, <c>expandFactor</c>, <c>convKernelSize</c> and
    /// <c>dtRank</c> all arrive in the constructor, and <c>GetParameters()</c> returns the full
    /// count immediately — 8,544 values for <c>MambaBlock&lt;float&gt;(4, 32, 8)</c>.
    /// </para>
    /// <para>
    /// Without this, <c>IsShapeResolved</c> stays false until a first input arrives and
    /// <see cref="LayerBase{T}.SetParameters"/> treats the block as shape-DEFERRED: a wrong-length
    /// vector is parked as a pending restore instead of being rejected. Loading mismatched weights
    /// then fails silently at construction and surfaces much later, somewhere unrelated.
    /// </para>
    /// </remarks>
    protected override bool ParametersAreConstructionSized => true;

    /// <summary>
    /// Gets the model dimension (d_model) of this Mamba block.
    /// </summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the SSM state dimension (N) controlling the capacity of the hidden state.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The state dimension controls how much "memory" the model has.
    /// A larger state can capture more complex patterns but uses more computation.
    /// Typical values are 16 (Mamba default) or 64 for higher capacity.
    /// </para>
    /// </remarks>
    public int StateDimension => _stateDimension;

    /// <summary>
    /// Gets the inner dimension (d_inner = modelDim * expandFactor) used for the SSM computation.
    /// </summary>
    public int InnerDimension => _innerDimension;

    /// <summary>
    /// Gets the convolution kernel size used in the depthwise Conv1D.
    /// </summary>
    public int ConvKernelSize => _convKernelSize;

    /// <summary>
    /// Gets the rank of the delta (dt) projection, which controls the low-rank bottleneck.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The delta timestep controls how much each input position
    /// influences the state update. Using a low-rank projection reduces parameters while
    /// maintaining expressivity. Default is ceil(modelDim / 16) following the original paper.
    /// </para>
    /// </remarks>
    public int DtRank => _dtRank;

    /// <summary>
    /// Creates a new Mamba block.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The width of the representation at each sequence position.
    /// Larger values give the model more capacity but use more memory.</para>
    /// </param>
    /// <param name="stateDimension">
    /// SSM state dimension (N). Default: 16.
    /// <para><b>For Beginners:</b> Controls the "memory capacity" of the state space model.
    /// The original Mamba paper uses N=16. Larger values (e.g., 64) increase capacity at the
    /// cost of more computation.</para>
    /// </param>
    /// <param name="expandFactor">
    /// Expansion factor for inner dimension. Default: 2.
    /// <para><b>For Beginners:</b> The SSM operates in an expanded dimension (modelDim * expandFactor)
    /// for more capacity, similar to the FFN expansion in Transformers. The original paper uses 2.</para>
    /// </param>
    /// <param name="convKernelSize">
    /// Convolution kernel size. Default: 4.
    /// <para><b>For Beginners:</b> The Conv1D captures short-range local patterns before the SSM
    /// processes the sequence. Kernel size 4 means each position sees 3 previous positions.</para>
    /// </param>
    /// <param name="dtRank">
    /// Rank of the delta projection. Default: -1 (auto = ceil(modelDim / 16)).
    /// <para><b>For Beginners:</b> Controls the bottleneck dimension for the timestep parameter.
    /// Using -1 lets the model auto-compute it following the paper's recommendation.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when modelDimension or stateDimension is not positive.</exception>
    public MambaBlock(
        int sequenceLength,
        int modelDimension = 256,
        int stateDimension = 16,
        int expandFactor = 2,
        int convKernelSize = 4,
        int dtRank = -1,
        IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(
            // Sequence is a FREE axis: -1, not the configured maximum. sequenceLength is
            // documented as a MAXIMUM and is used here for nothing but validation -- no weight and
            // no buffer is sized against it, because the recurrence runs over whatever length it
            // is handed. Publishing it as a concrete contract made the layer claim an output it
            // does not produce for any other length, which VerifyReportedOutputShape reports as
            // "[maxLen, D] declared but [B, actualLen, D] produced" and which anything sizing
            // itself from the declaration -- parameter slicing, chain resolution, ONNX export --
            // reads as fact. modelDimension IS structural and stays concrete.
            [-1, modelDimension],
            [-1, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        InitializationStrategy = initializationStrategy ?? InitializationStrategies<T>.Eager;

        if (sequenceLength <= 0)
        {
            throw new ArgumentException(
                $"Sequence length ({sequenceLength}) must be positive.", nameof(sequenceLength));
        }

        if (modelDimension <= 0)
        {
            throw new ArgumentException(
                $"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));
        }

        if (stateDimension <= 0)
        {
            throw new ArgumentException(
                $"State dimension ({stateDimension}) must be positive.", nameof(stateDimension));
        }

        if (expandFactor <= 0)
        {
            throw new ArgumentException(
                $"Expand factor ({expandFactor}) must be positive.", nameof(expandFactor));
        }

        if (convKernelSize <= 0)
        {
            throw new ArgumentException(
                $"Conv kernel size ({convKernelSize}) must be positive.", nameof(convKernelSize));
        }

        _modelDimension = modelDimension;
        _stateDimension = stateDimension;
        _innerDimension = modelDimension * expandFactor;
        _convKernelSize = convKernelSize;
        _dtRank = dtRank < 0 ? (int)Math.Ceiling((double)modelDimension / 16) : dtRank;

        // Input projection: [modelDim, innerDim * 2] (x branch + z branch)
        _inputProjectionWeights = new Tensor<T>([modelDimension, _innerDimension * 2]);
        _inputProjectionBias = new Tensor<T>([_innerDimension * 2]);

        // Depthwise Conv1D: [innerDim, convKernelSize]
        _convWeights = new Tensor<T>([_innerDimension, convKernelSize]);
        _convBias = new Tensor<T>([_innerDimension]);

        // x_proj: projects from innerDim to (dtRank + stateDim + stateDim) for delta, B, C
        _xProjectionWeights = new Tensor<T>([_innerDimension, _dtRank + stateDimension * 2]);

        // dt_proj: projects from dtRank to innerDim
        _dtProjectionWeights = new Tensor<T>([_dtRank, _innerDimension]);
        _dtProjectionBias = new Tensor<T>([_innerDimension]);

        // A_log: [innerDim, stateDim] (stored as log for numerical stability, A = -exp(A_log))
        _aLog = new Tensor<T>([_innerDimension, stateDimension]);

        // D: [innerDim] (skip connection)
        _dParam = new Tensor<T>([_innerDimension]);

        // Output projection: [innerDim, modelDim]
        _outputProjectionWeights = new Tensor<T>([_innerDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        // Xavier initialization for projection weights
        InitializeTensor(_inputProjectionWeights);
        _inputProjectionBias.Fill(NumOps.Zero);

        // Kaiming initialization for Conv1D (fan_in = convKernelSize)
        InitializeTensor(_convWeights);
        _convBias.Fill(NumOps.Zero);

        // Xavier for SSM projections
        InitializeTensor(_xProjectionWeights);
        InitializeTensor(_dtProjectionWeights);

        // Initialize dt bias with small positive values (ensures initial delta > 0 after softplus)
        for (int i = 0; i < _dtProjectionBias.Length; i++)
        {
            _dtProjectionBias[i] = NumOps.FromDouble(0.01);
        }

        // Initialize A_log: log of the S4D-Lin initialization
        // A = -exp(A_log) where A_log[d, n] = log(n + 1)
        // This gives the structured spacing from the S4D paper (Gu et al., 2022)
        for (int d = 0; d < _innerDimension; d++)
        {
            for (int n = 0; n < _stateDimension; n++)
            {
                _aLog[new[] { d, n }] = NumOps.FromDouble(Math.Log(n + 1));
            }
        }

        // D initialized to ones (skip connection)
        _dParam.Fill(NumOps.One);

        // Xavier for output projection
        InitializeTensor(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);

        // Register ALL trainable parameters for tape-based autodiff at construction time. The tape training
        // path (NeuralNetworkBase.Train -> TrainWithTape) collects registered parameters BEFORE the first
        // UpdateParameters call, so registering here (not only inside UpdateParameters) is what lets the
        // optimizer actually see and update this block's weights — otherwise CollectParameters finds nothing
        // and every Train step is a silent no-op. _aLog and _dParam are learnable SSM parameters (Gu & Dao
        // 2023) and MUST be registered too, or they would be excluded from gradient updates and make the
        // registered-vs-flat parameter counts disagree.
        RegisterTrainableParameters();
    }

    // Registers every trainable tensor with the autodiff/optimizer machinery. Called at init and re-called
    // after UpdateParameters replaces tensor instances so the registry always points at the live tensors.
    private void RegisterTrainableParameters()
    {
        RegisterTrainableParameter(_inputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_inputProjectionBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_convWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_convBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_xProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_dtProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_dtProjectionBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_aLog, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_dParam, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionBias, PersistentTensorRole.Biases);
    }

    private void InitializeTensor(Tensor<T> tensor)
    {
        InitializeLayerWeights(tensor, tensor.Shape[0], tensor.Shape.Length > 1 ? tensor.Shape[1] : tensor.Shape[0]);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        _originalInputShape = input._shape;

        int rank = input.Shape.Length;
        int seqLen = rank >= 2 ? input.Shape[rank - 2] : 1;
        int modelDim = input.Shape[rank - 1];

        // Flatten to 3D [batch, seq, modelDim]
        int batchSize = 1;
        for (int d = 0; d < rank - 2; d++)
            batchSize *= input.Shape[d];
        if (rank < 3) batchSize = 1;

        var input3D = rank == 2
            ? Engine.Reshape(input, new[] { 1, seqLen, modelDim })
            : Engine.Reshape(input, new[] { batchSize, seqLen, modelDim });

        _lastInput = input3D;

        // Step 1: Input projection -> x branch and z branch
        var input2D = Engine.Reshape(input3D, new[] { batchSize * seqLen, modelDim });
        var projected = Engine.TensorMatMul(input2D, _inputProjectionWeights);
        var bias2D = Engine.Reshape(_inputProjectionBias, new[] { 1, _innerDimension * 2 });
        var projectedWithBias = Engine.TensorAdd(projected, bias2D);
        var projected3D = Engine.Reshape(projectedWithBias, new[] { batchSize, seqLen, _innerDimension * 2 });

        // Split into x and z branches
        var xBranch = Engine.TensorNarrow(projected3D, 2, 0, _innerDimension);
        var zBranch = Engine.TensorNarrow(projected3D, 2, _innerDimension, _innerDimension);

        _lastXBranch = xBranch;
        _lastZBranch = zBranch;

        // Step 2: Conv1D on x branch (depthwise, causal) - Engine-accelerated
        var convOutput = DepthwiseConv1DForward(xBranch, seqLen);
        _lastConvOutput = convOutput;

        // Step 3: SiLU activation via Engine
        var siluOutput = Engine.Swish(convOutput);
        _lastSiluOutput = siluOutput;

        // Step 4: Project to SSM parameters (delta, B, C)
        var siluFlat = Engine.Reshape(siluOutput, new[] { batchSize * seqLen, _innerDimension });
        var xProj = Engine.TensorMatMul(siluFlat, _xProjectionWeights);
        var xProj3D = Engine.Reshape(xProj, new[] { batchSize, seqLen, _dtRank + _stateDimension * 2 });

        var deltaLowRank = Engine.TensorNarrow(xProj3D, 2, 0, _dtRank);
        var bParam = Engine.TensorNarrow(xProj3D, 2, _dtRank, _stateDimension);
        var cParam = Engine.TensorNarrow(xProj3D, 2, _dtRank + _stateDimension, _stateDimension);

        // Step 5: Project delta from low rank to inner dimension and apply softplus
        var deltaFlat = Engine.Reshape(deltaLowRank, new[] { batchSize * seqLen, _dtRank });
        var deltaProjFlat = Engine.TensorMatMul(deltaFlat, _dtProjectionWeights);
        var dtBias2D = Engine.Reshape(_dtProjectionBias, new[] { 1, _innerDimension });
        var deltaProjWithBias = Engine.TensorAdd(deltaProjFlat, dtBias2D);
        var deltaProj3D = Engine.Reshape(deltaProjWithBias, new[] { batchSize, seqLen, _innerDimension });

        _lastDeltaPreSoftplus = deltaProj3D;
        var delta = Engine.Softplus(deltaProj3D);
        _lastDelta = delta;
        _lastB = bParam;
        _lastC = cParam;

        // Step 6: Selective scan (core SSM computation).
        // Fast path (no carried initial state AND caller doesn't need state output):
        // use the engine's fused MambaSelectiveScanForward — a single tape op with
        // an exact BPTT backward (AiDotNet.Tensors#523/#1464). It replaces S6Scan's
        // per-timestep micro-op loop, which records O(seqLen) tape nodes and is the
        // dominant Mamba cost — catastrophically so in double precision and at the
        // long sequences 3D/vision Mamba models produce (e.g. SegMamba's 8^3 = 512
        // tokens). The decomposed S6Scan path is retained for two cases:
        //   1) a non-zero initial hidden state must be threaded across calls
        //      (stateful inference from a previous chunk), OR
        //   2) the caller will read GetHiddenState() after the forward — chunked
        //      autoregressive inference relies on this even when starting from
        //      zero state. Without it, _lastHiddenStates = null would leave the
        //      caller with no carry to feed into the next chunk.
        bool needsStateOutput = RequireHiddenStateOutput;
        Tensor<T> scanOutput;
        if (_initialHiddenState is null && !needsStateOutput)
        {
            scanOutput = Engine.MambaSelectiveScanForward(
                siluOutput, delta, _aLog, bParam, cParam, _dParam);
            _lastHiddenStates = null;
        }
        else
        {
            var (so, hiddenStatesResult) = S6Scan<T>.SequentialScanForward(
                siluOutput, delta, _aLog, bParam, cParam, _dParam,
                batchSize, seqLen, _innerDimension, _stateDimension,
                _initialHiddenState);
            scanOutput = so;
            _lastHiddenStates = hiddenStatesResult;
        }
        _initialHiddenState = null; // consumed
        _lastScanOutput = scanOutput;

        // Step 7: Output gating: y = scan_output * SiLU(z) via Engine
        var zGate = Engine.Swish(zBranch);
        var gatedOutput = Engine.TensorMultiply(scanOutput, zGate);
        _lastGatedOutput = gatedOutput;

        // Step 8: Output projection
        var gatedFlat = Engine.Reshape(gatedOutput, new[] { batchSize * seqLen, _innerDimension });
        var outputFlat = Engine.TensorMatMul(gatedFlat, _outputProjectionWeights);
        var outBias2D = Engine.Reshape(_outputProjectionBias, new[] { 1, _modelDimension });
        var outputWithBias = Engine.TensorAdd(outputFlat, outBias2D);
        var output3D = Engine.Reshape(outputWithBias, new[] { batchSize, seqLen, _modelDimension });

        // Residual connection: h = h + Block(h). Standard Mamba block pattern
        // per Gu & Dao 2023 (state-spaces/mamba reference impl wraps the inner
        // block in `residual + Block(LN(residual))`). Without it, repeated
        // block stacks attenuate ~3 orders of magnitude per layer (observed
        // 0.036 → 1e-38 through 4 blocks in MultiLayerModel_ProducesNonTrivialOutput),
        // collapsing the LM head's logits to zero and producing a uniform
        // distribution. input3D and output3D both have shape
        // [batchSize, seqLen, _modelDimension] so the add is shape-aligned.
        var residualOutput = Engine.TensorAdd(output3D, input3D);

        var result = ApplyActivation(residualOutput);
        _lastOutput = result;

        // Reshape back to original rank
        if (rank == 2)
            return Engine.Reshape(result, new[] { seqLen, _modelDimension });

        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
            outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen;
        outputShape[rank - 1] = _modelDimension;
        return Engine.Reshape(result, outputShape);
    }

    #region Engine-Accelerated Conv1D

    /// <summary>
    /// Depthwise causal Conv1D using Engine tensor operations.
    /// </summary>
    /// <remarks>
    /// Uses the engine's depthwise Conv1D primitive so the entire causal convolution is one
    /// differentiable operation instead of expanding <c>sequenceLength * kernelSize</c> narrow,
    /// multiply, and add nodes onto the tape. The public Mamba parameter layout stores coefficients
    /// as <c>[channel, lag]</c>, where lag zero is the current token. The NCL convolution primitive
    /// performs cross-correlation, so reverse the lag axis and use <c>kernelSize - 1</c> symmetric
    /// padding, then retain the first <c>sequenceLength</c> positions. This is exactly equivalent to
    /// <c>sum_k weight[channel,k] * input[t-k,channel]</c> and preserves the paper's causal contract.
    /// </remarks>
    private Tensor<T> DepthwiseConv1DForward(Tensor<T> input, int seqLen)
    {
        var inputNcl = Engine.TensorPermute(input, new[] { 0, 2, 1 }).Contiguous();
        var reversedWeights = Engine.TensorFlip(_convWeights, new[] { 1 });
        var kernel = Engine.Reshape(reversedWeights, new[] { _innerDimension, 1, _convKernelSize });
        var padded = Engine.DepthwiseConv1D(
            inputNcl, kernel, stride: 1, padding: _convKernelSize - 1);
        var causalNcl = Engine.TensorNarrow(padded, dim: 2, start: 0, length: seqLen);
        var causal = Engine.TensorPermute(causalNcl, new[] { 0, 2, 1 }).Contiguous();
        var bias = Engine.Reshape(_convBias, new[] { 1, 1, _innerDimension });
        return Engine.TensorAdd(causal, bias);
    }

    /// <summary>
    /// Backward pass for depthwise causal Conv1D using explicit per-element computation.
    /// Avoids Engine 3D tensor operations that have contiguity/reduction bugs.
    /// </summary>
    private Tensor<T> DepthwiseConv1DBackward(
        Tensor<T> dOutput, Tensor<T> input, int batchSize, int seqLen)
    {
        var dInput = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _innerDimension });
        // Zero-initialize rented buffer — it may contain stale data from previous use
        for (int i = 0; i < dInput.Length; i++) dInput[i] = NumOps.Zero;
        _convBiasGradient = new Tensor<T>(new[] { _innerDimension });
        _convWeightsGradient = new Tensor<T>(new[] { _innerDimension, _convKernelSize });

        // Depthwise conv1d: output[b,t,d] = sum_k(weight[d,k] * input[b,t-k,d]) + bias[d]
        // Backward:
        //   dInput[b,srcT,d] += weight[d,k] * dOutput[b,t,d]  where srcT = t-k
        //   dWeight[d,k] += sum_b(input[b,srcT,d] * dOutput[b,t,d])
        //   dBias[d] += sum_b,t(dOutput[b,t,d])
        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                for (int d = 0; d < _innerDimension; d++)
                {
                    T dOutVal = dOutput[new[] { bi, t, d }];

                    // Bias gradient
                    _convBiasGradient[d] = NumOps.Add(_convBiasGradient[d], dOutVal);

                    for (int k = 0; k < _convKernelSize; k++)
                    {
                        int srcT = t - k;
                        if (srcT >= 0)
                        {
                            T w = _convWeights[new[] { d, k }];
                            T xVal = input[new[] { bi, srcT, d }];

                            // Input gradient
                            dInput[new[] { bi, srcT, d }] = NumOps.Add(
                                dInput[new[] { bi, srcT, d }],
                                NumOps.Multiply(w, dOutVal));

                            // Weight gradient
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

    #region Engine-Accelerated Helpers

    /// <summary>
    /// Workaround for Engine.ReduceSum multi-axis [0,1] bug (AiDotNet.Tensors PR #62).
    /// Sums a [batch, seq, features] tensor over batch and seq → [features].
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

    /// <summary>
    /// Computes SiLU derivative using Engine operations: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)).
    /// </summary>
    private Tensor<T> ComputeSiLUDerivative(Tensor<T> input)
    {
        var sig = Engine.Sigmoid(input);
        var negSig = Engine.TensorNegate(sig);
        var oneMinusSig = Engine.TensorAddScalar(negSig, NumOps.One);
        var xSigOneMinusSig = Engine.TensorMultiply(input,
            Engine.TensorMultiply(sig, oneMinusSig));
        return Engine.TensorAdd(sig, xSigOneMinusSig);
    }

    #endregion

    #region Tensor Manipulation Helpers

    /// <summary>
    /// Slices a tensor along a given axis, extracting a contiguous range.
    /// </summary>
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
        _xProjectionWeights = Engine.TensorAdd(_xProjectionWeights, Engine.TensorMultiplyScalar(_xProjectionWeightsGradient!, negLR));
        _dtProjectionWeights = Engine.TensorAdd(_dtProjectionWeights, Engine.TensorMultiplyScalar(_dtProjectionWeightsGradient!, negLR));
        _dtProjectionBias = Engine.TensorAdd(_dtProjectionBias, Engine.TensorMultiplyScalar(_dtProjectionBiasGradient!, negLR));
        _aLog = Engine.TensorAdd(_aLog, Engine.TensorMultiplyScalar(_aLogGradient!, negLR));
        _dParam = Engine.TensorAdd(_dParam, Engine.TensorMultiplyScalar(_dParamGradient!, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(_outputProjectionWeightsGradient!, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(_outputProjectionBiasGradient!, negLR));

        // Re-register against the new tensor instances created by the updates above (TensorAdd returns new
        // tensors), so the autodiff registry tracks the live weights — now including _aLog and _dParam.
        RegisterTrainableParameters();
    }

    public override Vector<T> GetParameterGradients()
    {
        if (_inputProjectionWeightsGradient == null) return new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));
        return Vector<T>.Concatenate(
            new Vector<T>(_inputProjectionWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_inputProjectionBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_convWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_convBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_xProjectionWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_dtProjectionWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_dtProjectionBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_aLogGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_dParamGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_outputProjectionWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_outputProjectionBiasGradient?.ToArray() ?? Array.Empty<T>()));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _inputProjectionWeightsGradient = null; _inputProjectionBiasGradient = null;
        _convWeightsGradient = null; _convBiasGradient = null;
        _xProjectionWeightsGradient = null;
        _dtProjectionWeightsGradient = null; _dtProjectionBiasGradient = null;
        _aLogGradient = null; _dParamGradient = null;
        _outputProjectionWeightsGradient = null; _outputProjectionBiasGradient = null;
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
        _lastScanOutput = null;
        _lastGatedOutput = null;
        _lastDelta = null;
        _lastDeltaPreSoftplus = null;
        _lastB = null;
        _lastC = null;
        _lastHiddenStates = null;
        _originalInputShape = null;
        _inputProjectionWeightsGradient = null;
        _inputProjectionBiasGradient = null;
        _convWeightsGradient = null;
        _convBiasGradient = null;
        _xProjectionWeightsGradient = null;
        _dtProjectionWeightsGradient = null;
        _dtProjectionBiasGradient = null;
        _aLogGradient = null;
        _dParamGradient = null;
        _outputProjectionWeightsGradient = null;
        _outputProjectionBiasGradient = null;
    }

    #endregion

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["StateDimension"] = _stateDimension.ToString();
        metadata["InnerDimension"] = _innerDimension.ToString();
        metadata["ConvKernelSize"] = _convKernelSize.ToString();
        metadata["DtRank"] = _dtRank.ToString();
        // Publish the CONSTRUCTOR argument, not just the derived width. Reconstruction calls the
        // ctor, whose parameter is expandFactor; it cannot use InnerDimension. Without this key the
        // rebuilt block silently fell back to the ctor default of 2, so any model configured with a
        // different factor came back double-width and rejected its own saved parameters
        // ("Expected 40920 parameters, got 20472" restoring a TimeMachine, whose factor is 1).
        metadata["ExpandFactor"] = (_modelDimension > 0 ? _innerDimension / _modelDimension : 1).ToString();
        return metadata;
    }

    /// <summary>
    /// Gets a copy of the input projection weights for external inspection or quantization.
    /// </summary>
    public Tensor<T> GetInputProjectionWeights() => _inputProjectionWeights.Clone();

    /// <summary>
    /// Gets a copy of the output projection weights for external inspection or quantization.
    /// </summary>
    public Tensor<T> GetOutputProjectionWeights() => _outputProjectionWeights.Clone();

    /// <summary>
    /// Gets a copy of the A_log parameter tensor (A = -exp(A_log)) for external inspection.
    /// </summary>
    public Tensor<T> GetALogParameter() => _aLog.Clone();

    /// <summary>
    /// Gets a copy of the D skip connection parameter for external inspection.
    /// </summary>
    public Tensor<T> GetDParameter() => _dParam.Clone();

    /// <summary>
    /// Overwrites the D (skip-connection) parameter in place.
    /// </summary>
    /// <param name="values">The replacement values; must match the current D length.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="values"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the length does not match D.</exception>
    /// <remarks>
    /// <see cref="GetDParameter"/> returns a CLONE, so a caller holding it cannot write back through
    /// it. Quantization needs to: D is the residual skip term in Gu and Dao's selective SSM, and
    /// rounding it to 4 bits changes the layer's identity path, which is why
    /// <c>SSMQuantizationHelper.QuantizeSSMLayer</c> offers to protect it.
    ///
    /// That protection previously worked by hand-computing D's offset into the flat parameter
    /// vector from the layer's dimensions. The arithmetic did not match the registry's actual
    /// ordering, so it restored the wrong slice and left D quantized anyway (1 came back as
    /// 0.94967109). Going through the tensor removes the offset arithmetic entirely.
    /// </remarks>
    public void SetDParameter(Tensor<T> values)
    {
        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (values.Length != _dParam.Length)
        {
            throw new ArgumentException(
                $"D has {_dParam.Length} values but {values.Length} were supplied.", nameof(values));
        }

        for (int i = 0; i < _dParam.Length; i++)
        {
            _dParam[i] = values[i];
        }
    }

    /// <summary>
    /// Gets the current hidden state from the last forward pass, if available.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returns the SSM hidden states stored during the most recent forward pass.
    /// The shape is [batch, seqLen+1, innerDim, stateDim] where index 0 is the initial
    /// (zero) state and subsequent indices are states after each timestep.
    /// </para>
    /// <para><b>For Beginners:</b> After processing a sequence, this returns the model's
    /// internal "memory" from each step. This is used by the state cache during autoregressive
    /// generation to avoid recomputing previous states.</para>
    /// </remarks>
    /// <returns>The hidden states tensor, or null if no forward pass has been performed.</returns>
    public Tensor<T>? GetHiddenState() => _lastHiddenStates;

    /// <summary>
    /// When true, the forward pass always routes through the decomposed S6Scan
    /// path so the per-step hidden states are available via <see cref="GetHiddenState"/>
    /// after the call. Set this to true on the encoder block of stateful / chunked
    /// inference pipelines that read the trailing hidden state to seed the next
    /// chunk; leave it false for end-to-end training and one-shot inference where
    /// the fused <c>MambaSelectiveScanForward</c> fast path is preferred.
    /// </summary>
    public bool RequireHiddenStateOutput { get; set; }

    /// <summary>
    /// Sets the initial hidden state for the next forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When set, the next call to <see cref="Forward"/> will start from this state instead of zeros.
    /// The state is consumed (reset to null) after one forward pass. Shape must be
    /// [batch, innerDim, stateDim].
    /// </para>
    /// <para><b>For Beginners:</b> This restores the model's "memory" from a previous step,
    /// allowing autoregressive generation to continue from where it left off instead of
    /// starting fresh each time.</para>
    /// </remarks>
    /// <param name="state">The hidden state tensor [batch, innerDim, stateDim].</param>
    /// <exception cref="ArgumentNullException">Thrown when state is null.</exception>
    /// <exception cref="ArgumentException">Thrown when state has wrong rank or dimensions.</exception>
    public void SetHiddenState(Tensor<T> state)
    {
        if (state == null)
            throw new ArgumentNullException(nameof(state));
        if (state.Rank != 3)
            throw new ArgumentException(
                $"Hidden state must be rank 3 [batch, innerDim, stateDim], but got rank {state.Rank}.", nameof(state));
        if (state.Shape[1] != _innerDimension)
            throw new ArgumentException(
                $"Hidden state dimension 1 must be {_innerDimension} (innerDim), but got {state.Shape[1]}.", nameof(state));
        if (state.Shape[2] != _stateDimension)
            throw new ArgumentException(
                $"Hidden state dimension 2 must be {_stateDimension} (stateDim), but got {state.Shape[2]}.", nameof(state));

        _initialHiddenState = state;
    }
}

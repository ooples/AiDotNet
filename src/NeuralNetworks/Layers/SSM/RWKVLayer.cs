using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Memory;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the RWKV (Receptance Weighted Key Value) layer, a linear attention RNN from Peng et al., 2024.
/// </summary>
/// <remarks>
/// <para>
/// RWKV combines the training parallelism of Transformers with the efficient inference of RNNs.
/// It uses a linear attention mechanism with data-dependent decay, avoiding the quadratic complexity
/// of standard attention while maintaining competitive quality for language modeling.
/// </para>
/// <para>
/// The architecture consists of two mixing modules per layer:
/// <code>
///   Time Mixing (attention replacement):
///     r_t = W_r * (mu_r * x_t + (1-mu_r) * x_{t-1})      // Receptance (gate)
///     k_t = W_k * (mu_k * x_t + (1-mu_k) * x_{t-1})      // Key
///     v_t = W_v * (mu_v * x_t + (1-mu_v) * x_{t-1})      // Value
///     w_t = W_w * x_t + b_w                                // Data-dependent decay (v6+)
///     wkv_t = (sum of exp(-(t-i)*w) * exp(k_i) * v_i) / (sum of exp(-(t-i)*w) * exp(k_i))
///     output_t = sigmoid(r_t) * wkv_t
///
///   Channel Mixing (FFN replacement):
///     r_t = W_r * (mu_r * x_t + (1-mu_r) * x_{t-1})
///     k_t = W_k * (mu_k * x_t + (1-mu_k) * x_{t-1})
///     output_t = sigmoid(r_t) * (W_v * max(k_t, 0)^2)     // Squared ReLU gating
/// </code>
/// </para>
/// <para>
/// RWKV v6 (Finch) adds data-dependent linear interpolation for the token-shift mixing coefficients,
/// making mu_r, mu_k, mu_v input-dependent rather than fixed learned parameters.
/// </para>
/// <para><b>For Beginners:</b> RWKV is like a clever hybrid between a Transformer and an RNN.
///
/// Imagine you're summarizing a conversation:
/// - A Transformer re-reads the entire conversation for every new sentence (expensive but thorough)
/// - An RNN keeps a running summary and just adds new info (cheap but may forget)
/// - RWKV keeps a running summary (like an RNN) but uses a smart weighting scheme
///   so recent information is weighted more heavily, like how you naturally pay more attention
///   to what was just said
///
/// The "token shift" mechanism is like looking at both the current word and the previous word
/// to decide what's important - a simple but effective trick.
///
/// Used by RWKV Foundation models (Eagle v5, Finch v6, Goose v7) which achieve competitive
/// performance with Transformers at much lower inference cost.
/// </para>
/// <para>
/// <b>Reference:</b> Peng et al., "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence", 2024.
/// https://arxiv.org/abs/2404.05892
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerCategory(LayerCategory.Recurrent)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, SupportsBackpropagation = false, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 256", TestConstructorArgs = "4")]
// SEQUENCE IS A FREE AXIS, and the constructor already says why at length: sequenceLength is a MAXIMUM
// used for validation only, no weight is sized against it, and publishing it as a concrete contract made
// the layer claim an output it does not produce for any other length. So Time is carried, never pinned -
// ForwardTraced writes "outputShape[rank - 2] = seqLen", the length it was handed.
//
// BatchOptional covers both accepted forms from one declaration: rank 2 is the [Time, Features] the
// layer's own [LayerProperty(TestInputShape = "4, 256")] exercises, rank 3 the batched form. Rank 1 is
// NOT declared, and that is a real limit rather than caution: at rank 1 the tail of ForwardTraced
// indexes outputShape[rank - 2], i.e. outputShape[-1]. Rank 4+ runs and carries its leading axes, but
// each would need a distinct role to be named and there is no second batch-like role to give it.
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
// The layer registered its 8 weight matrices and silently omitted 10 more LEARNED tensors --
// _timeMixR/K/V and _channelMixR/K (the mixing coefficients RWKV is named for), _bonus, and both
// LayerNorm affine pairs. The optimizer never updated them, so the layer only partly trained and
// nothing reported it.
//
// [AutoParameters] alone did NOT fix that, though the comment here used to say it did. That
// attribute is a migration MARKER; its own documentation states it "does not classify fields" and
// that the generator is driven by the per-field semantic declarations instead. So the ten fields
// stayed unregistered: absent from the tape training path, which trains only from registered
// parameters, and absent from the saved checkpoint. Measured by perturbing each field and reading
// it back after a round trip -- all ten discarded a 0.25 change while _decayBias, the one field
// that already carried [TrainableParameter], kept it exactly. They now carry the attribute too.
//
// The output cannot show this. _outputWeights and _channelValueWeights are deliberately
// zero-initialized (see InitializeParameters) so the block is an identity residual at init, which
// multiplies every upstream parameter's influence by zero: a freshly built RWKVLayer returns its
// input bit-for-bit. Any check that compares outputs is therefore blind here and will report a
// clean round trip no matter what was dropped.
[AutoParameters]
public partial class RWKVLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Hand-written for two reasons. The generator keys its arms on the declared layout length and does
    /// not expand <c>BatchOptional</c>, so it would cover rank 3 and decline the rank 2 this layer is
    /// tested at; and the trailing axis is <c>Fixed</c>, not <c>Same</c>, which the generator cannot
    /// derive from roles alone.
    /// </para>
    /// <para>
    /// <c>Fixed(_modelDimension)</c> is what the code actually does - <c>ForwardTraced</c> writes
    /// <c>outputShape[rank - 1] = _modelDimension</c>, the field, not <c>input.Shape[rank - 1]</c>. The
    /// two coincide for any input the layer accepts, because both residual adds
    /// (<c>TensorAdd(input3D, timeMixOut)</c> and <c>TensorAdd(afterTimeMix, channelMixOut)</c>) require
    /// the incoming width to equal the model dimension. Declaring the field is the stronger and more
    /// honest of the two readings: it says the width is a property of the LAYER, which is what makes a
    /// mismatched input a shape error rather than something inference should propagate.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank is not (2 or 3) || _modelDimension <= 0) return null;

        var time = new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time));
        var features = new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_modelDimension));

        return inputRank == 2
            ? new[] { time, features }
            : new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                time, features,
            };
    }

    // Configuration
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;

    // Time mixing parameters
    [TrainableParameter(Role = PersistentTensorRole.ScaleParameters)]
    private Tensor<T> _timeMixR;  // [modelDim] lerp coefficient for receptance
    [TrainableParameter(Role = PersistentTensorRole.ScaleParameters)]
    private Tensor<T> _timeMixK;  // [modelDim] lerp coefficient for key
    [TrainableParameter(Role = PersistentTensorRole.ScaleParameters)]
    private Tensor<T> _timeMixV;  // [modelDim] lerp coefficient for value

    // Time mixing projections: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _receptanceWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _keyWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _valueWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputWeights;

    // RWKV-4 time_decay (w): LEARNED STATIC per-channel decay [modelDim]; effective decay = -exp(w).
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _decayBias;     // [modelDim] — RWKV-4 time_decay

    // RWKV-4 time_first (u): per-channel current-token bonus [numHeads, headDim] == [modelDim].
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _bonus;

    // Channel mixing parameters
    [TrainableParameter(Role = PersistentTensorRole.ScaleParameters)]
    private Tensor<T> _channelMixR;  // [modelDim]
    [TrainableParameter(Role = PersistentTensorRole.ScaleParameters)]
    private Tensor<T> _channelMixK;  // [modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _channelKeyWeights;    // [modelDim, modelDim * 4]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _channelValueWeights;  // [modelDim * 4, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _channelReceptanceWeights;  // [modelDim, modelDim]

    // Layer norm parameters
    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]
    private Tensor<T> _normGamma1;
    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]
    private Tensor<T> _normBeta1;
    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]
    private Tensor<T> _normGamma2;
    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]
    private Tensor<T> _normBeta2;

    // Cached values for backward pass
    [Scratch]
    private Tensor<T>? _lastInput;
    [Scratch]
    private Tensor<T>? _lastOutput;
    [Scratch]
    private Tensor<T>? _lastTimeMixOutput;
    [Scratch]
    private Tensor<T>? _lastChannelMixOutput;
    [Scratch]
    private Tensor<T>? _lastReceptance;
    [Scratch]
    private Tensor<T>? _lastWkv;
    [Scratch]
    private Tensor<T>? _lastState;
    private int[]? _originalInputShape;

    // Gradients
    [Scratch]
    private Tensor<T>? _timeMixRGradient;
    [Scratch]
    private Tensor<T>? _timeMixKGradient;
    [Scratch]
    private Tensor<T>? _timeMixVGradient;
    [Scratch]
    private Tensor<T>? _receptanceWeightsGradient;
    [Scratch]
    private Tensor<T>? _keyWeightsGradient;
    [Scratch]
    private Tensor<T>? _valueWeightsGradient;
    [Scratch]
    private Tensor<T>? _outputWeightsGradient;
    [Scratch]
    private Tensor<T>? _decayBiasGradient;
    [Scratch]
    private Tensor<T>? _bonusGradient;
    [Scratch]
    private Tensor<T>? _channelMixRGradient;
    [Scratch]
    private Tensor<T>? _channelMixKGradient;
    [Scratch]
    private Tensor<T>? _channelKeyWeightsGradient;
    [Scratch]
    private Tensor<T>? _channelValueWeightsGradient;
    [Scratch]
    private Tensor<T>? _channelReceptanceWeightsGradient;
    [Scratch]
    private Tensor<T>? _normGamma1Gradient;
    [Scratch]
    private Tensor<T>? _normBeta1Gradient;
    [Scratch]
    private Tensor<T>? _normGamma2Gradient;
    [Scratch]
    private Tensor<T>? _normBeta2Gradient;

    /// <inheritdoc />
    /// <summary>
    /// Training IS supported. The forward pass (time-mixing WKV recurrence, decay, channel-mixing,
    /// token-shift, and all projections) is expressed entirely in tape-connected engine ops, so
    /// gradients for every parameter flow through the autodiff tape (issue #1464). Previously the
    /// recurrence ran in detached scalar code, which forced SupportsTraining=false and diverged
    /// training in the RWKV-4/5/6 language models via a residual-only gradient mismatch.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Every weight is sized from constructor arguments, so the parameter surface is known before
    /// the first forward pass.
    /// </summary>
    /// <remarks>
    /// <c>GetParameters()</c> already returns 13,664 values for <c>RWKVLayer<float>(4, 32, 4)</c>. Without this,
    /// <c>IsShapeResolved</c> stays false and <see cref="LayerBase{T}.SetParameters"/> treats the
    /// layer as shape-DEFERRED, parking a wrong-length vector as a pending restore instead of
    /// rejecting it -- so mismatched weights fail silently and surface later somewhere unrelated.
    /// </remarks>
    protected override bool ParametersAreConstructionSized => true;

    /// <summary>
    /// Gets the model dimension.
    /// </summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the number of attention heads.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets the dimension per head.
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>Construction state: the 'sequenceLength' the layer was built with.</summary>
    private readonly int _sequenceLength;

    /// <summary>
    /// Creates a new RWKV layer.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> Width of the representation at each position. RWKV models range
    /// from 169M (d=768) to 14B (d=5120) parameters.</para>
    /// </param>
    /// <param name="numHeads">
    /// Number of heads for matrix-valued states. Default: 8.
    /// <para><b>For Beginners:</b> RWKV v5+ uses multi-headed states, similar to multi-head attention.
    /// Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public RWKVLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
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
        _sequenceLength = sequenceLength;
        InitializationStrategy = initializationStrategy ?? InitializationStrategies<T>.Eager;

        if (sequenceLength <= 0)
            throw new ArgumentException($"Sequence length ({sequenceLength}) must be positive.", nameof(sequenceLength));
        if (modelDimension <= 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));
        if (numHeads <= 0)
            throw new ArgumentException($"Number of heads ({numHeads}) must be positive.", nameof(numHeads));
        if (modelDimension % numHeads != 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;

        int expandedDim = modelDimension * 4;

        // Time mixing
        _timeMixR = new Tensor<T>([modelDimension]);
        _timeMixK = new Tensor<T>([modelDimension]);
        _timeMixV = new Tensor<T>([modelDimension]);
        _receptanceWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputWeights = new Tensor<T>([modelDimension, modelDimension]);
        _decayBias = new Tensor<T>([modelDimension]);
        _bonus = new Tensor<T>([numHeads, _headDimension]);

        // Channel mixing
        _channelMixR = new Tensor<T>([modelDimension]);
        _channelMixK = new Tensor<T>([modelDimension]);
        _channelKeyWeights = new Tensor<T>([modelDimension, expandedDim]);
        _channelValueWeights = new Tensor<T>([expandedDim, modelDimension]);
        _channelReceptanceWeights = new Tensor<T>([modelDimension, modelDimension]);

        // Layer norms
        _normGamma1 = new Tensor<T>([modelDimension]);
        _normBeta1 = new Tensor<T>([modelDimension]);
        _normGamma2 = new Tensor<T>([modelDimension]);
        _normBeta2 = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        // Token shift mixing coefficients: initialized to 0.5 (equal mix of current and previous)
        for (int i = 0; i < _modelDimension; i++)
        {
            T halfVal = NumOps.FromDouble(0.5);
            _timeMixR[i] = halfVal;
            _timeMixK[i] = halfVal;
            _timeMixV[i] = halfVal;
            _channelMixR[i] = halfVal;
            _channelMixK[i] = halfVal;
        }

        InitializeTensor(_receptanceWeights);
        InitializeTensor(_keyWeights);
        InitializeTensor(_valueWeights);
        // RWKV init: the projections that write INTO the residual stream — the time-mix output
        // (_outputWeights) and the channel-mix value (_channelValueWeights) — are ZERO-initialized
        // (left at the new-tensor zero) so every block starts as an identity residual. This keeps a
        // deep stack numerically stable at init (no compounding noise across layers) and is what
        // lets RWKV train without the gradient explosion a non-zero output init causes.
        _decayBias.Fill(NumOps.FromDouble(-5.0));  // RWKV-4 time_decay init; effective decay = -exp(-5) ≈ -0.0067
        _bonus.Fill(NumOps.FromDouble(0.5));  // Small bonus for current token

        InitializeTensor(_channelKeyWeights);
        InitializeTensor(_channelReceptanceWeights);

        _normGamma1.Fill(NumOps.One);
        _normBeta1.Fill(NumOps.Zero);
        _normGamma2.Fill(NumOps.One);
        _normBeta2.Fill(NumOps.Zero);
    }

    private void InitializeTensor(Tensor<T> tensor)
    {
        InitializeLayerWeights(tensor, tensor.Shape[0], tensor.Shape[1]);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
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

        // Time mixing sub-layer (with residual)
        var normed1 = ApplyLayerNorm(input3D, _normGamma1, _normBeta1, batchSize, seqLen);
        var timeMixOut = TimeMixingForward(normed1, batchSize, seqLen);
        _lastTimeMixOutput = timeMixOut;
        var afterTimeMix = Engine.TensorAdd(input3D, timeMixOut);

        // Channel mixing sub-layer (with residual)
        var normed2 = ApplyLayerNorm(afterTimeMix, _normGamma2, _normBeta2, batchSize, seqLen);
        var channelMixOut = ChannelMixingForward(normed2, batchSize, seqLen);
        _lastChannelMixOutput = channelMixOut;
        var output3D = Engine.TensorAdd(afterTimeMix, channelMixOut);

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
    /// Time mixing forward: token shift + linear attention with exponential decay.
    /// </summary>
    private Tensor<T> TimeMixingForward(Tensor<T> x, int batchSize, int seqLen)
    {
        // ---- #1464 + trainability: the ENTIRE time-mixing path is expressed in tape-connected
        // Engine ops so gradients flow through the WKV recurrence (decay/key/value/receptance
        // projections, the token-shift mix coefficients, and the output projection all train).
        // Previously the recurrence was computed in detached scalar NumOps/Math.Exp code, which left
        // SupportsTraining=false and — because the residual skip still carried the input gradient —
        // produced a gradient MISMATCH that diverged training in the RWKV-4/5/6 language models. The
        // token-shift + projections are batched over the whole sequence (one GEMM each); only the
        // matrix-state recurrence is sequential. Vectorized over bh = batch*numHeads, headDim.
        int bsl = batchSize * seqLen;

        var x3 = Engine.Reshape(x, new[] { batchSize, seqLen, _modelDimension });
        var ones1D = Tensor<T>.CreateDefault(new[] { _modelDimension }, NumOps.One);
        var mixR3 = Engine.Reshape(_timeMixR, new[] { 1, 1, _modelDimension });
        var mixK3 = Engine.Reshape(_timeMixK, new[] { 1, 1, _modelDimension });
        var mixV3 = Engine.Reshape(_timeMixV, new[] { 1, 1, _modelDimension });
        var invR3 = Engine.Reshape(Engine.TensorSubtract(ones1D, _timeMixR), new[] { 1, 1, _modelDimension });
        var invK3 = Engine.Reshape(Engine.TensorSubtract(ones1D, _timeMixK), new[] { 1, 1, _modelDimension });
        var invV3 = Engine.Reshape(Engine.TensorSubtract(ones1D, _timeMixV), new[] { 1, 1, _modelDimension });
        var xPrev0 = new Tensor<T>(new[] { batchSize, 1, _modelDimension });
        var xShifted = seqLen > 1
            ? Engine.TensorConcatenate(new[] { xPrev0, Engine.TensorNarrow(x3, 1, 0, seqLen - 1) }, axis: 1)
            : xPrev0;

        var rInAll = Engine.TensorAdd(Engine.TensorMultiply(x3, mixR3), Engine.TensorMultiply(xShifted, invR3));
        var kInAll = Engine.TensorAdd(Engine.TensorMultiply(x3, mixK3), Engine.TensorMultiply(xShifted, invK3));
        var vInAll = Engine.TensorAdd(Engine.TensorMultiply(x3, mixV3), Engine.TensorMultiply(xShifted, invV3));

        var Rall = Engine.Reshape(Engine.TensorMatMul(Engine.Reshape(rInAll, new[] { bsl, _modelDimension }), _receptanceWeights), new[] { batchSize, seqLen, _modelDimension });
        var Kall = Engine.Reshape(Engine.TensorMatMul(Engine.Reshape(kInAll, new[] { bsl, _modelDimension }), _keyWeights), new[] { batchSize, seqLen, _modelDimension });
        var Vall = Engine.Reshape(Engine.TensorMatMul(Engine.Reshape(vInAll, new[] { bsl, _modelDimension }), _valueWeights), new[] { batchSize, seqLen, _modelDimension });

        // Paper-faithful RWKV-4 WKV (Peng et al. 2023, the official numerically-stable kernel):
        // per-CHANNEL scalar state (aa = weighted value sum, bb = weight sum) with a running max (pp)
        // so the exponentials can never overflow. time_decay (w) and time_first (u) are LEARNED
        // STATIC per-channel parameters (input-independent — the data-dependent decay matmul that made
        // the old code an unfaithful v6/amalgam, and the matrix num/den state, are both gone):
        //   ww = u + k_t; q = max(pp, ww); wkv = (e^{pp-q}·aa + e^{ww-q}·v) / (e^{pp-q}·bb + e^{ww-q})
        //   out = sigmoid(r)·wkv
        //   ww2 = pp + w; q2 = max(ww2, k); aa = e^{ww2-q2}·aa + e^{k-q2}·v; bb = e^{ww2-q2}·bb + e^{k-q2}; pp = q2
        // Every exp argument is <= 0 (the running max is subtracted), so no clamping is needed and
        // training is numerically stable. All ops are tape-connected, so every parameter trains.
        var u = Engine.Reshape(_bonus, new[] { 1, _modelDimension });                                            // time_first
        var w = Engine.TensorNegate(Engine.TensorExp(Engine.Reshape(_decayBias, new[] { 1, _modelDimension }))); // -exp(time_decay) < 0

        // Per-channel state (channels = modelDim), broadcast over the batch.
        var aa = new Tensor<T>(new[] { batchSize, _modelDimension });
        var bb = new Tensor<T>(new[] { batchSize, _modelDimension });
        var pp = Tensor<T>.CreateDefault(new[] { batchSize, _modelDimension }, NumOps.FromDouble(-1e38));
        var outputSlices = new System.Collections.Generic.List<Tensor<T>>(seqLen);

        for (int t = 0; t < seqLen; t++)
        {
            // RECORDED slice via Engine.TensorNarrow. Wrapping a BARE
            // GetSliceAlongDimension view in Engine.Reshape is NOT sufficient: the Reshape is a
            // recorded op but its INPUT is still a non-owning view that no compiled step produces,
            // so the fused compiled-training plan captures it as a graph LEAF frozen at trace time
            // and every timestep then reads abandoned trace-time storage (zeros first, recycled
            // garbage later). It also means the slice never joined the gradient tape, so the
            // upstream weights received no gradient through it.
            // Same defect and same fix as RealGatedLinearRecurrenceLayer — see the detailed
            // rationale there (#1789 Generated Q-S). RWKV4's failures were the identical signature:
            // Clone_AfterTraining / MoreData / OptimizerStep_ParamL2.
            var k_t = Engine.Reshape(Engine.TensorNarrow(Kall, 1, t, 1), new[] { batchSize, _modelDimension });
            var v_t = Engine.Reshape(Engine.TensorNarrow(Vall, 1, t, 1), new[] { batchSize, _modelDimension });
            var r_t = Engine.Reshape(Engine.TensorNarrow(Rall, 1, t, 1), new[] { batchSize, _modelDimension });

            // Output for this token (current key boosted by the time_first bonus u).
            var ww = Engine.TensorAdd(k_t, u);
            var q = Engine.TensorMax(pp, ww);
            var e1 = Engine.TensorExp(Engine.TensorSubtract(pp, q));
            var e2 = Engine.TensorExp(Engine.TensorSubtract(ww, q));
            var wkv = Engine.TensorDivide(
                Engine.TensorAdd(Engine.TensorMultiply(e1, aa), Engine.TensorMultiply(e2, v_t)),
                Engine.TensorAdd(Engine.TensorMultiply(e1, bb), e2));
            outputSlices.Add(Engine.Reshape(Engine.TensorMultiply(Engine.Sigmoid(r_t), wkv), new[] { batchSize, 1, _modelDimension }));

            // State update with the static time-decay w (no bonus on the carried state).
            var ww2 = Engine.TensorAdd(pp, w);
            var q2 = Engine.TensorMax(ww2, k_t);
            var e1b = Engine.TensorExp(Engine.TensorSubtract(ww2, q2));
            var e2b = Engine.TensorExp(Engine.TensorSubtract(k_t, q2));
            aa = Engine.TensorAdd(Engine.TensorMultiply(e1b, aa), Engine.TensorMultiply(e2b, v_t));
            bb = Engine.TensorAdd(Engine.TensorMultiply(e1b, bb), e2b);
            pp = q2;
        }

        // Tape-connected assembly + batched output projection so the WKV path + output weights train.
        var wkvAll = seqLen > 0
            ? Engine.TensorConcatenate(outputSlices.ToArray(), axis: 1)
            : new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var output = Engine.Reshape(
            Engine.TensorMatMul(Engine.Reshape(wkvAll, new[] { bsl, _modelDimension }), _outputWeights),
            new[] { batchSize, seqLen, _modelDimension });

        _lastState = seqLen > 0
            ? x.GetSliceAlongDimension(seqLen - 1, 1).Clone()
            : new Tensor<T>(new[] { batchSize, _modelDimension });
        _lastReceptance = output;  // Cache for backward
        _lastWkv = output;
        return output;
    }

    /// <summary>
    /// Channel mixing forward: squared ReLU with receptance gating.
    /// </summary>
    private Tensor<T> ChannelMixingForward(Tensor<T> x, int batchSize, int seqLen)
    {
        int bsl = batchSize * seqLen;

        // ---- #1464 + trainability: channel mixing is purely position-wise (token-shift + a
        // squared-ReLU FFN, no recurrence), so the whole sub-layer is batched over the sequence —
        // one GEMM each — and kept fully tape-connected so the channel-mix projection weights and
        // mix coefficients train. xShifted[:, t, :] = x[:, t-1, :], t=0 -> zeros.
        var x3 = Engine.Reshape(x, new[] { batchSize, seqLen, _modelDimension });
        var ones1D = Tensor<T>.CreateDefault(new[] { _modelDimension }, NumOps.One);
        var mixR3 = Engine.Reshape(_channelMixR, new[] { 1, 1, _modelDimension });
        var mixK3 = Engine.Reshape(_channelMixK, new[] { 1, 1, _modelDimension });
        var invR3 = Engine.Reshape(Engine.TensorSubtract(ones1D, _channelMixR), new[] { 1, 1, _modelDimension });
        var invK3 = Engine.Reshape(Engine.TensorSubtract(ones1D, _channelMixK), new[] { 1, 1, _modelDimension });
        var xPrev0 = new Tensor<T>(new[] { batchSize, 1, _modelDimension });
        var xShifted = seqLen > 1
            ? Engine.TensorConcatenate(new[] { xPrev0, Engine.TensorNarrow(x3, 1, 0, seqLen - 1) }, axis: 1)
            : xPrev0;
        var rIn = Engine.Reshape(
            Engine.TensorAdd(Engine.TensorMultiply(x3, mixR3), Engine.TensorMultiply(xShifted, invR3)),
            new[] { bsl, _modelDimension });
        var kIn = Engine.Reshape(
            Engine.TensorAdd(Engine.TensorMultiply(x3, mixK3), Engine.TensorMultiply(xShifted, invK3)),
            new[] { bsl, _modelDimension });

        // r = sigmoid(W_r · rIn); k = W_k · kIn; squared-ReLU = ReLU(k)^2; v = W_v · kSq; out = sigmoid(r)·v.
        var rGate = Engine.Sigmoid(Engine.TensorMatMul(rIn, _channelReceptanceWeights)); // [bsl, modelDim]
        var kProj = Engine.TensorMatMul(kIn, _channelKeyWeights);                        // [bsl, expandedDim]
        var kRelu = Engine.ReLU(kProj);
        var kSquared = Engine.TensorMultiply(kRelu, kRelu);                              // tape-connected max(0,k)^2
        var vProj = Engine.TensorMatMul(kSquared, _channelValueWeights);                 // [bsl, modelDim]
        var output = Engine.Reshape(Engine.TensorMultiply(rGate, vProj), new[] { batchSize, seqLen, _modelDimension });
        return output;
    }

    /// <summary>
    /// Applies layer normalization.
    /// </summary>
    private Tensor<T> ApplyLayerNorm(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta,
        int batchSize, int seqLen)
    {
        var shaped = input.Reshape([batchSize, seqLen, _modelDimension]);
        return Engine.LayerNorm(shaped, gamma, beta, 1e-6, out _, out _);
    }

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_timeMixRGradient is null || _timeMixKGradient is null || _timeMixVGradient is null ||
            _receptanceWeightsGradient is null || _keyWeightsGradient is null ||
            _valueWeightsGradient is null || _outputWeightsGradient is null ||
            _decayBiasGradient is null || _bonusGradient is null ||
            _channelMixRGradient is null || _channelMixKGradient is null ||
            _channelKeyWeightsGradient is null || _channelValueWeightsGradient is null ||
            _channelReceptanceWeightsGradient is null ||
            _normGamma1Gradient is null || _normBeta1Gradient is null ||
            _normGamma2Gradient is null || _normBeta2Gradient is null)
        {
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        }

        T negLR = NumOps.Negate(learningRate);
        _timeMixR = Engine.TensorAdd(_timeMixR, Engine.TensorMultiplyScalar(_timeMixRGradient, negLR));
        _timeMixK = Engine.TensorAdd(_timeMixK, Engine.TensorMultiplyScalar(_timeMixKGradient, negLR));
        _timeMixV = Engine.TensorAdd(_timeMixV, Engine.TensorMultiplyScalar(_timeMixVGradient, negLR));
        _receptanceWeights = Engine.TensorAdd(_receptanceWeights, Engine.TensorMultiplyScalar(_receptanceWeightsGradient, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGradient, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGradient, negLR));
        _outputWeights = Engine.TensorAdd(_outputWeights, Engine.TensorMultiplyScalar(_outputWeightsGradient, negLR));
        _decayBias = Engine.TensorAdd(_decayBias, Engine.TensorMultiplyScalar(_decayBiasGradient, negLR));
        _bonus = Engine.TensorAdd(_bonus, Engine.TensorMultiplyScalar(_bonusGradient, negLR));
        _channelMixR = Engine.TensorAdd(_channelMixR, Engine.TensorMultiplyScalar(_channelMixRGradient, negLR));
        _channelMixK = Engine.TensorAdd(_channelMixK, Engine.TensorMultiplyScalar(_channelMixKGradient, negLR));
        _channelKeyWeights = Engine.TensorAdd(_channelKeyWeights, Engine.TensorMultiplyScalar(_channelKeyWeightsGradient, negLR));
        _channelValueWeights = Engine.TensorAdd(_channelValueWeights, Engine.TensorMultiplyScalar(_channelValueWeightsGradient, negLR));
        _channelReceptanceWeights = Engine.TensorAdd(_channelReceptanceWeights, Engine.TensorMultiplyScalar(_channelReceptanceWeightsGradient, negLR));
        _normGamma1 = Engine.TensorAdd(_normGamma1, Engine.TensorMultiplyScalar(_normGamma1Gradient, negLR));
        _normBeta1 = Engine.TensorAdd(_normBeta1, Engine.TensorMultiplyScalar(_normBeta1Gradient, negLR));
        _normGamma2 = Engine.TensorAdd(_normGamma2, Engine.TensorMultiplyScalar(_normGamma2Gradient, negLR));
        _normBeta2 = Engine.TensorAdd(_normBeta2, Engine.TensorMultiplyScalar(_normBeta2Gradient, negLR));

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_receptanceWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_decayBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_channelKeyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_channelValueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_channelReceptanceWeights, PersistentTensorRole.Weights);

    }

    private Tensor<T>[] GetAllTensors() =>
    [
        _timeMixR, _timeMixK, _timeMixV,
        _receptanceWeights, _keyWeights, _valueWeights, _outputWeights,
        _decayBias, _bonus,
        _channelMixR, _channelMixK,
        _channelKeyWeights, _channelValueWeights, _channelReceptanceWeights,
        _normGamma1, _normBeta1, _normGamma2, _normBeta2
    ];

    public override Vector<T> GetParameterGradients()
    {
        if (_timeMixRGradient == null) return new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));
        return Vector<T>.Concatenate(
            new Vector<T>(_timeMixRGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_timeMixKGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_timeMixVGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_receptanceWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_keyWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_valueWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_outputWeightsGradient?.ToArray() ?? new T[_outputWeights.Length]),
            new Vector<T>(_decayBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_bonusGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_channelMixRGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_channelMixKGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_channelKeyWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_channelValueWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_channelReceptanceWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_normGamma1Gradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_normBeta1Gradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_normGamma2Gradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_normBeta2Gradient?.ToArray() ?? Array.Empty<T>()));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _timeMixRGradient = null; _timeMixKGradient = null; _timeMixVGradient = null; _receptanceWeightsGradient = null; _keyWeightsGradient = null; _valueWeightsGradient = null; _outputWeightsGradient = null; _decayBiasGradient = null; _bonusGradient = null; _channelMixRGradient = null; _channelMixKGradient = null; _channelKeyWeightsGradient = null; _channelValueWeightsGradient = null; _channelReceptanceWeightsGradient = null; _normGamma1Gradient = null; _normBeta1Gradient = null; _normGamma2Gradient = null; _normBeta2Gradient = null;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastTimeMixOutput = null;
        _lastChannelMixOutput = null;
        _lastReceptance = null;
        _lastWkv = null;
        _lastState = null;
        _originalInputShape = null;
        _timeMixRGradient = null;
        _timeMixKGradient = null;
        _timeMixVGradient = null;
        _receptanceWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _outputWeightsGradient = null;
        _decayBiasGradient = null;
        _bonusGradient = null;
        _channelMixRGradient = null;
        _channelMixKGradient = null;
        _channelKeyWeightsGradient = null;
        _channelValueWeightsGradient = null;
        _channelReceptanceWeightsGradient = null;
        _normGamma1Gradient = null;
        _normBeta1Gradient = null;
        _normGamma2Gradient = null;
        _normBeta2Gradient = null;
    }

    #endregion

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["NumHeads"] = _numHeads.ToString();
        metadata["HeadDimension"] = _headDimension.ToString();
        return metadata;
    }

    /// <summary>
    /// Gets a copy of the receptance projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetReceptanceWeights() => _receptanceWeights.Clone();

    /// <summary>
    /// Gets a copy of the output projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetOutputWeights() => _outputWeights.Clone();
}

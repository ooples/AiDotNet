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
public partial class RWKVLayer<T> : LayerBase<T>
{
    // Configuration
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;

    // Time mixing parameters
    private Tensor<T> _timeMixR;  // [modelDim] lerp coefficient for receptance
    private Tensor<T> _timeMixK;  // [modelDim] lerp coefficient for key
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
    private Tensor<T> _bonus;

    // Channel mixing parameters
    private Tensor<T> _channelMixR;  // [modelDim]
    private Tensor<T> _channelMixK;  // [modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _channelKeyWeights;    // [modelDim, modelDim * 4]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _channelValueWeights;  // [modelDim * 4, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _channelReceptanceWeights;  // [modelDim, modelDim]

    // Layer norm parameters
    private Tensor<T> _normGamma1;
    private Tensor<T> _normBeta1;
    private Tensor<T> _normGamma2;
    private Tensor<T> _normBeta2;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastTimeMixOutput;
    private Tensor<T>? _lastChannelMixOutput;
    private Tensor<T>? _lastReceptance;
    private Tensor<T>? _lastWkv;
    private Tensor<T>? _lastState;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _timeMixRGradient;
    private Tensor<T>? _timeMixKGradient;
    private Tensor<T>? _timeMixVGradient;
    private Tensor<T>? _receptanceWeightsGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _outputWeightsGradient;
    private Tensor<T>? _decayBiasGradient;
    private Tensor<T>? _bonusGradient;
    private Tensor<T>? _channelMixRGradient;
    private Tensor<T>? _channelMixKGradient;
    private Tensor<T>? _channelKeyWeightsGradient;
    private Tensor<T>? _channelValueWeightsGradient;
    private Tensor<T>? _channelReceptanceWeightsGradient;
    private Tensor<T>? _normGamma1Gradient;
    private Tensor<T>? _normBeta1Gradient;
    private Tensor<T>? _normGamma2Gradient;
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

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override long ParameterCount =>
        _timeMixR.Length + _timeMixK.Length + _timeMixV.Length +
        _receptanceWeights.Length + _keyWeights.Length + _valueWeights.Length + _outputWeights.Length +
        _decayBias.Length + _bonus.Length +
        _channelMixR.Length + _channelMixK.Length +
        _channelKeyWeights.Length + _channelValueWeights.Length + _channelReceptanceWeights.Length +
        _normGamma1.Length + _normBeta1.Length + _normGamma2.Length + _normBeta2.Length;

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
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
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
        InitializeTensor(_outputWeights);
        _decayBias.Fill(NumOps.FromDouble(-5.0));  // RWKV-4 time_decay init; effective decay = -exp(-5) ≈ -0.0067
        _bonus.Fill(NumOps.FromDouble(0.5));  // Small bonus for current token

        InitializeTensor(_channelKeyWeights);
        InitializeTensor(_channelValueWeights);
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

        var rInAll = Engine.TensorAdd(Engine.TensorBroadcastMultiply(x3, mixR3), Engine.TensorBroadcastMultiply(xShifted, invR3));
        var kInAll = Engine.TensorAdd(Engine.TensorBroadcastMultiply(x3, mixK3), Engine.TensorBroadcastMultiply(xShifted, invK3));
        var vInAll = Engine.TensorAdd(Engine.TensorBroadcastMultiply(x3, mixV3), Engine.TensorBroadcastMultiply(xShifted, invV3));

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
            var k_t = Engine.Reshape(Kall.GetSliceAlongDimension(t, 1), new[] { batchSize, _modelDimension });
            var v_t = Engine.Reshape(Vall.GetSliceAlongDimension(t, 1), new[] { batchSize, _modelDimension });
            var r_t = Engine.Reshape(Rall.GetSliceAlongDimension(t, 1), new[] { batchSize, _modelDimension });

            // Output for this token (current key boosted by the time_first bonus u).
            var ww = Engine.TensorBroadcastAdd(k_t, u);
            var q = Engine.TensorMax(pp, ww);
            var e1 = Engine.TensorExp(Engine.TensorSubtract(pp, q));
            var e2 = Engine.TensorExp(Engine.TensorSubtract(ww, q));
            var wkv = Engine.TensorDivide(
                Engine.TensorAdd(Engine.TensorMultiply(e1, aa), Engine.TensorMultiply(e2, v_t)),
                Engine.TensorAdd(Engine.TensorMultiply(e1, bb), e2));
            outputSlices.Add(Engine.Reshape(Engine.TensorMultiply(Engine.Sigmoid(r_t), wkv), new[] { batchSize, 1, _modelDimension }));

            // State update with the static time-decay w (no bonus on the carried state).
            var ww2 = Engine.TensorBroadcastAdd(pp, w);
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
            Engine.TensorAdd(Engine.TensorBroadcastMultiply(x3, mixR3), Engine.TensorBroadcastMultiply(xShifted, invR3)),
            new[] { bsl, _modelDimension });
        var kIn = Engine.Reshape(
            Engine.TensorAdd(Engine.TensorBroadcastMultiply(x3, mixK3), Engine.TensorBroadcastMultiply(xShifted, invK3)),
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

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        int totalParams = ParameterCountHelper.ToFlatVectorSize(ParameterCount);
        var parameters = new Vector<T>(totalParams);
        int index = 0;

        foreach (var tensor in GetAllTensors())
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
        foreach (var tensor in GetAllTensors())
        {
            for (int i = 0; i < tensor.Length; i++)
                tensor[i] = parameters[index++];
        }
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

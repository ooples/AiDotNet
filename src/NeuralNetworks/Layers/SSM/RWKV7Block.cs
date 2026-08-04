using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Memory;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements a single RWKV-7 "Goose" block with the WKV-7 kernel featuring dynamic state evolution.
/// </summary>
/// <remarks>
/// <para>
/// RWKV-7 is the seventh generation of the RWKV architecture, introducing expressive dynamic state
/// evolution that replaces the fixed exponential decay of previous versions with learnable, data-dependent
/// transition matrices. This allows the model to dynamically control how information is stored, retained,
/// and forgotten in the recurrent state.
/// </para>
/// <para>
/// Each block contains two sub-layers with residual connections:
/// <code>
///   Time Mixing (WKV-7 kernel):
///     1. Token shift: lerp between current and previous token
///     2. Compute r, k, v, a, b from shifted inputs via linear projections
///     3. WKV-7 state update: state_t = diag(a_t) * state_{t-1} + b_t^T * (k_t * v_t^T)
///     4. Output: sigmoid(r_t) * GroupNorm(state_t * k_t)
///     5. Linear output projection
///
///   Channel Mixing (SiLU gating):
///     1. Token shift: lerp between current and previous token
///     2. r_t = W_r * shifted_r, k_t = W_k * shifted_k
///     3. output = sigmoid(r_t) * (W_v * SiLU(k_t))
/// </code>
/// </para>
/// <para>
/// Key innovations over RWKV-6 "Finch":
/// <list type="bullet">
///   <item>Learnable transition vectors a_t (diagonal state decay) and b_t (additive state injection)</item>
///   <item>State evolution: S_t = diag(a_t) * S_{t-1} + b_t * (k_t * v_t), replacing fixed exp decay</item>
///   <item>Group normalization on WKV output for stability</item>
///   <item>SiLU activation in channel mixing instead of squared ReLU</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> This is one layer in the RWKV-7 model. Think of it as a smart
/// information processor that:
/// 1. Reads the current word and blends it with the previous word
/// 2. Decides what to remember and what to forget (using learnable transition rules)
/// 3. Produces an output that captures both local and long-range context
///
/// Unlike Transformers that re-read the entire text each time, RWKV-7 keeps a compact running
/// summary (the "state") that gets updated with each new word, making it very efficient.
/// </para>
/// <para>
/// <b>Reference:</b> Peng et al., "RWKV-7 Goose with Expressive Dynamic State Evolution", 2025.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerCategory(LayerCategory.Recurrent)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 16", TestConstructorArgs = "4, 16, 2")]
public partial class RWKV7Block<T> : LayerBase<T>
{
    // Configuration
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly int _ffnDimension;

    // ============ Time Mixing Parameters ============

    // Token shift mixing coefficients: [modelDim]
    private Tensor<T> _timeMixR;
    private Tensor<T> _timeMixK;
    private Tensor<T> _timeMixV;
    private Tensor<T> _timeMixA;  // v7: shift coefficient for 'a' (state decay)
    private Tensor<T> _timeMixB;  // v7: shift coefficient for 'b' (state injection)

    // Linear projections: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _receptanceWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _keyWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _valueWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputWeights;

    // v7: Dynamic state evolution projections
    /// <summary>
    /// Low-rank decay projection, first factor: <c>w1</c> in <c>w = w0 + tanh(x_w @ w1) @ w2</c>.
    /// </summary>
    /// <remarks>
    /// The reference (RWKV-LM RWKV-v7) makes the data-dependent part of the decay both LOW-RANK and
    /// tanh-BOUNDED, and initialises this factor to ZERO so the decay starts exactly at the w0 ramp
    /// and the projection only earns influence through training. A full-rank unbounded projection
    /// initialised to noise — which is what this layer used to have — perturbs the ramp from the
    /// first step and lets the decay logit drift without limit, which is how a state with
    /// near-1.0 retention channels runs away.
    /// </remarks>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _w1;        // [modelDim, decayLoraRank], zeros

    /// <summary>Low-rank decay projection, second factor: <c>w2</c>. Orthogonal init, gain 0.1.</summary>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _w2;        // [decayLoraRank, modelDim]

    /// <summary>
    /// Low-rank ICL-rate projection, first factor: <c>a1</c> in <c>a = sigmoid(a0 + (x_a @ a1) @ a2)</c>.
    /// Zero-initialised for the same reason as <see cref="_w1"/>; note the reference applies NO tanh
    /// on this path, the sigmoid alone bounds it.
    /// </summary>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _a1;        // [modelDim, iclLoraRank], zeros

    /// <summary>Low-rank ICL-rate projection, second factor: <c>a2</c>. Orthogonal init, gain 0.1.</summary>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _a2;        // [iclLoraRank, modelDim]

    /// <summary>
    /// LoRA rank shared by the decay and ICL-rate projections:
    /// <c>max(32, round(2.5*sqrt(C)/32)*32)</c>, per the reference.
    /// </summary>
    private readonly int _loraRank;

    /// <summary>Gate LoRA rank: <c>max(32, round(5*sqrt(C)/32)*32)</c> — wider than the decay/ICL rank.</summary>
    private readonly int _gateLoraRank;

    /// <summary>Value-residual LoRA rank: <c>max(32, round(1.7*sqrt(C)/32)*32)</c> — the narrowest of the three.</summary>
    private readonly int _mvLoraRank;

    /// <summary>Bias of the value-residual gate, <c>v0</c>. Init <c>0.73 - linear*0.4</c>.</summary>
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _v0;        // [modelDim]

    /// <summary>Value-residual LoRA first factor, zero-initialised.</summary>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _v1;        // [modelDim, mvLoraRank]

    /// <summary>Value-residual LoRA second factor. Orthogonal init, gain 0.1.</summary>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _v2;        // [mvLoraRank, modelDim]

    /// <summary>
    /// The value-residual gate parameters (v0, v1, v2), in that order.
    /// </summary>
    /// <remarks>
    /// Exposed so the cross-layer gradient tests can target exactly these. They are the only
    /// parameters whose gradient depends on a DIFFERENT layer's output, so they are the ones a
    /// broken cross-layer edge silently starves.
    /// </remarks>
    internal Tensor<T>[] ValueResidualParameters => [_v0, _v1, _v2];

    /// <summary>The value projection weights, which the first layer publishes as v_first.</summary>
    internal Tensor<T> ValueProjectionWeights => _valueWeights;

    /// <summary>
    /// The time-mixing output projection, which RWKV-7 initializes to exactly zero.
    /// </summary>
    /// <remarks>
    /// Exposed for gradient tests, which must move it off zero first. While it is zero the whole
    /// time-mixing branch contributes nothing, so <c>dL/dW = normed^T (dL/dout) W_out^T = 0</c> for
    /// EVERY parameter upstream of it. That zero is correct, not a defect — but it means a gradient
    /// test run at initialization measures nothing at all.
    /// </remarks>
    internal Tensor<T> OutputProjectionWeights => _outputWeights;

    /// <summary>v_first handed in for THIS forward pass; null when this block is the first layer.</summary>
    private Tensor<T>? _incomingVFirst;

    /// <summary>v_first this block publishes for the next one. Per-pass, never carried across calls.</summary>
    private Tensor<T>? _publishedVFirst;

    /// <summary>
    /// Per-head, per-channel bonus scale: <c>x += (r (*) k (*) r_k).sum(-1) * v</c>, applied AFTER
    /// the group norm (arXiv:2503.14456).
    /// </summary>
    /// <remarks>
    /// A direct current-token path that bypasses the recurrent state entirely — the head's own r·k
    /// agreement, scaled per channel, gates a copy of v straight into the output. Omitting it does
    /// not break shapes, so it compiles and trains while quietly removing one of the two routes
    /// information can take through the block. Reference init: zeros(H, N) - 0.04.
    /// </remarks>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _rk;        // [numHeads, headDim]

    /// <summary>Token-shift mix for the gate branch, <c>x_g</c> in the reference.</summary>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _timeMixG;  // [modelDim]

    /// <summary>Gate LoRA first factor, zero-initialised: <c>g = sigmoid(x_g @ g1) @ g2</c>.</summary>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _g1;        // [modelDim, gateLoraRank]

    /// <summary>Gate LoRA second factor. Orthogonal init, gain 0.1.</summary>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _g2;        // [gateLoraRank, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _aBias;     // [modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _bBias;     // [modelDim]

    /// <summary>
    /// Per-channel scale forming the removal key: kappa_t = k_t (*) k_k (arXiv:2503.14456, Eq. 17).
    /// </summary>
    /// <remarks>
    /// The kernel L2-normalises kappa per head, so only this vector's DIRECTION per head matters to
    /// the removal term; its magnitude is divided out. Reference init (RWKV-LM RWKV-v7):
    /// <c>k_k = 0.71 - linear*0.1</c> with <c>linear[n] = n/(C-1) - 0.5</c>.
    /// </remarks>
    private Tensor<T> _kk;        // [modelDim]

    /// <summary>
    /// Per-channel scale forming the value-injection key: kTilde_t = k_t (*) (1 + (a_t - 1) (*) k_a).
    /// </summary>
    /// <remarks>
    /// Interpolates the key toward its ICL-rate-modulated form. At a_t = 1 this is the identity
    /// (kTilde = k), so the block degrades to a plain delta rule when the in-context learning rate
    /// saturates. Reference init: <c>k_a = 1.02</c>.
    /// </remarks>
    private Tensor<T> _ka;        // [modelDim]

    // v7: Group norm on WKV output (per head)
    private Tensor<T> _groupNormGamma;  // [modelDim]
    private Tensor<T> _groupNormBeta;   // [modelDim]

    // ============ Channel Mixing Parameters ============

    private Tensor<T> _channelMixR;  // [modelDim]
    private Tensor<T> _channelMixK;  // [modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _channelKeyWeights;        // [modelDim, ffnDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _channelValueWeights;      // [ffnDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _channelReceptanceWeights; // [modelDim, modelDim]

    // ============ Layer Norms ============

    private Tensor<T> _normGamma1;
    private Tensor<T> _normBeta1;
    private Tensor<T> _normGamma2;
    private Tensor<T> _normBeta2;

    // ============ Cached values for backward ============

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastTimeMixOutput;
    private Tensor<T>? _lastChannelMixOutput;
    private Tensor<T>? _lastNormed1;
    private Tensor<T>? _lastNormed2;
    private Tensor<T>? _lastAfterTimeMix;
    private int[]? _originalInputShape;

    // Time mixing forward caches
    private Tensor<T>? _cachedWkvOut;
    private Tensor<T>? _cachedR;       // [batch, seqLen, modelDim] receptance projection
    private Tensor<T>? _cachedK;       // [batch, seqLen, modelDim] key projection
    private Tensor<T>? _cachedV;       // [batch, seqLen, modelDim] value projection
    private Tensor<T>? _cachedTimeMixNormed1; // [batch, seqLen, modelDim] normed input to time mixing
    // WKV pre-gate values are reconstructed from _cachedWkvGated / sigmoid(r) during backward
    private Tensor<T>? _cachedWkvGated;    // [batch, seqLen, modelDim] after gate, before groupNorm
    // Previous tokens per timestep reconstructed from _cachedTimeMixNormed1 during backward

    // Channel mixing forward caches
    private Tensor<T>? _cachedChannelRGate;   // [batch, seqLen, modelDim] sigmoid(W_r * rInput)
    private Tensor<T>? _cachedChannelSiLU;    // [batch, seqLen, ffnDim] SiLU(W_k * kInput)
    private Tensor<T>? _cachedChannelVProj;   // [batch, seqLen, modelDim] W_v * SiLU(k)

    // ============ Gradients ============

    private Tensor<T>? _timeMixRGrad;
    private Tensor<T>? _timeMixKGrad;
    private Tensor<T>? _timeMixVGrad;
    private Tensor<T>? _timeMixAGrad;
    private Tensor<T>? _timeMixBGrad;
    private Tensor<T>? _receptanceWeightsGrad;
    private Tensor<T>? _keyWeightsGrad;
    private Tensor<T>? _valueWeightsGrad;
    private Tensor<T>? _outputWeightsGrad;
    private Tensor<T>? _w1Grad;
    private Tensor<T>? _w2Grad;
    private Tensor<T>? _aBiasGrad;
    private Tensor<T>? _a1Grad;
    private Tensor<T>? _a2Grad;
    private Tensor<T>? _bBiasGrad;
    // Must exist and sit at the same index as _kk/_ka in GetAllParameterTensors: the two lists are
    // zipped positionally by UpdateParameters and GetParameterGradients.
    private Tensor<T>? _v0Grad;
    private Tensor<T>? _v1Grad;
    private Tensor<T>? _v2Grad;
    private Tensor<T>? _rkGrad;
    private Tensor<T>? _timeMixGGrad;
    private Tensor<T>? _g1Grad;
    private Tensor<T>? _g2Grad;
    private Tensor<T>? _kkGrad;
    private Tensor<T>? _kaGrad;
    private Tensor<T>? _groupNormGammaGrad;
    private Tensor<T>? _groupNormBetaGrad;
    private Tensor<T>? _channelMixRGrad;
    private Tensor<T>? _channelMixKGrad;
    private Tensor<T>? _channelKeyWeightsGrad;
    private Tensor<T>? _channelValueWeightsGrad;
    private Tensor<T>? _channelReceptanceWeightsGrad;
    private Tensor<T>? _normGamma1Grad;
    private Tensor<T>? _normBeta1Grad;
    private Tensor<T>? _normGamma2Grad;
    private Tensor<T>? _normBeta2Grad;

    // Recurrent state for autoregressive inference
    private Tensor<T>? _recurrentState;       // [batch, numHeads, headDim, headDim]
    private Tensor<T>? _prevToken;            // [batch, modelDim] for time mixing token shift
    private Tensor<T>? _prevChannelToken;     // [batch, modelDim] for channel mixing token shift

    /// <summary>
    /// Training support is approximate: gradients flow through residual connections and weight
    /// gradients are accumulated, but full backpropagation through the WKV-7 recurrent kernel
    /// is not yet implemented. Suitable for fine-tuning with small learning rates.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>Gets the model dimension.</summary>
    public int ModelDimension => _modelDimension;

    /// <summary>Gets the number of attention heads.</summary>
    public int NumHeads => _numHeads;

    /// <summary>Gets the dimension per head.</summary>
    public int HeadDimension => _headDimension;

    /// <summary>Gets the feed-forward network dimension.</summary>
    public int FFNDimension => _ffnDimension;

    /// <summary>
    /// RWKV-7's Global ICLR Multiplier <c>c</c>, applied to the state transition's removal term.
    /// </summary>
    private readonly double _globalIclrMultiplier;

    /// <summary>
    /// The paper's clamping lower bound <c>u = exp(-e^(-1/2))</c> on the decay multiplier
    /// (arXiv:2503.14456, Eq. 12 and Appendix C, Theorem 1 — quoted there as 0.5452...).
    /// </summary>
    private static readonly double DecayClampLowerBound = Math.Exp(-Math.Exp(-0.5));

    /// <inheritdoc />
    public override long ParameterCount
    {
        get
        {
            int count = 0;
            foreach (var tensor in GetAllParameterTensors())
                count += tensor.Length;
            return count;
        }
    }

    /// <summary>
    /// Creates a new RWKV-7 block.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">Model dimension (d_model). Default: 256.</param>
    /// <param name="numHeads">Number of heads for multi-headed states. Default: 4. Must divide modelDimension.</param>
    /// <param name="ffnMultiplier">FFN expansion multiplier. Default: 3.5 (RWKV-7 standard).</param>
    /// <param name="activationFunction">Optional activation on final output.</param>
    public RWKV7Block(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 4,
        double ffnMultiplier = 3.5,
        IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null,
        double globalIclrMultiplier = 1.0)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        // Theorem 1 (arXiv 2503.14456, Appendix C) is stated for c in (0, 1 + u); outside that range
        // the eigenvalue bound it proves no longer applies, so reject rather than silently accept.
        if (globalIclrMultiplier <= 0.0 || globalIclrMultiplier >= 1.0 + DecayClampLowerBound)
            throw new ArgumentOutOfRangeException(
                nameof(globalIclrMultiplier),
                globalIclrMultiplier,
                $"Global ICLR multiplier must lie in (0, {1.0 + DecayClampLowerBound}).");
        _globalIclrMultiplier = globalIclrMultiplier;

        InitializationStrategy = initializationStrategy ?? InitializationStrategies<T>.Eager;

        if (sequenceLength <= 0)
            throw new ArgumentException($"Sequence length ({sequenceLength}) must be positive.", nameof(sequenceLength));
        if (modelDimension <= 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));
        if (numHeads <= 0)
            throw new ArgumentException($"Number of heads ({numHeads}) must be positive.", nameof(numHeads));
        if (modelDimension % numHeads != 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));
        if (ffnMultiplier <= 0)
            throw new ArgumentException($"FFN multiplier ({ffnMultiplier}) must be positive.", nameof(ffnMultiplier));

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;
        _ffnDimension = (int)(modelDimension * ffnMultiplier);

        // Time mixing
        _timeMixR = new Tensor<T>([modelDimension]);
        _timeMixK = new Tensor<T>([modelDimension]);
        _timeMixV = new Tensor<T>([modelDimension]);
        _timeMixA = new Tensor<T>([modelDimension]);
        _timeMixB = new Tensor<T>([modelDimension]);

        _receptanceWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputWeights = new Tensor<T>([modelDimension, modelDimension]);

        // LoRA rank shared by both projections: max(32, round(2.5*sqrt(C)/32)*32), per the reference.
        _loraRank = Math.Max(32, (int)Math.Round(2.5 * Math.Sqrt(modelDimension) / 32.0) * 32);

        _w1 = new Tensor<T>([modelDimension, _loraRank]);
        _w2 = new Tensor<T>([_loraRank, modelDimension]);
        _aBias = new Tensor<T>([modelDimension]);
        _a1 = new Tensor<T>([modelDimension, _loraRank]);
        _a2 = new Tensor<T>([_loraRank, modelDimension]);
        _bBias = new Tensor<T>([modelDimension]);
        _gateLoraRank = Math.Max(32, (int)Math.Round(5.0 * Math.Sqrt(modelDimension) / 32.0) * 32);
        _mvLoraRank = Math.Max(32, (int)Math.Round(1.7 * Math.Sqrt(modelDimension) / 32.0) * 32);
        _v0 = new Tensor<T>([modelDimension]);
        _v1 = new Tensor<T>([modelDimension, _mvLoraRank]);
        _v2 = new Tensor<T>([_mvLoraRank, modelDimension]);
        _rk = new Tensor<T>([numHeads, modelDimension / numHeads]);
        _timeMixG = new Tensor<T>([modelDimension]);
        _g1 = new Tensor<T>([modelDimension, _gateLoraRank]);
        _g2 = new Tensor<T>([_gateLoraRank, modelDimension]);
        _kk = new Tensor<T>([modelDimension]);
        _ka = new Tensor<T>([modelDimension]);

        _groupNormGamma = new Tensor<T>([modelDimension]);
        _groupNormBeta = new Tensor<T>([modelDimension]);

        // Channel mixing
        _channelMixR = new Tensor<T>([modelDimension]);
        _channelMixK = new Tensor<T>([modelDimension]);
        _channelKeyWeights = new Tensor<T>([modelDimension, _ffnDimension]);
        _channelValueWeights = new Tensor<T>([_ffnDimension, modelDimension]);
        _channelReceptanceWeights = new Tensor<T>([modelDimension, modelDimension]);

        // Layer norms
        _normGamma1 = new Tensor<T>([modelDimension]);
        _normBeta1 = new Tensor<T>([modelDimension]);
        _normGamma2 = new Tensor<T>([modelDimension]);
        _normBeta2 = new Tensor<T>([modelDimension]);

        InitializeParameters();

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_receptanceWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_w1, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_w2, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_aBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_a1, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_a2, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_v0, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_v1, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_v2, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_rk, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_timeMixG, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_g1, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_g2, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_kk, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_ka, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_channelKeyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_channelValueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_channelReceptanceWeights, PersistentTensorRole.Weights);

        // Token-shift interpolation (mu) vectors, group-norm and layer-norm affine
        // parameters are all learnable in RWKV-7. With the forward now fully
        // tape-connected they receive correct gradients, so register them too.
        RegisterTrainableParameter(_timeMixR, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_timeMixK, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_timeMixV, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_timeMixA, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_timeMixB, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_channelMixR, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_channelMixK, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_groupNormGamma, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_groupNormBeta, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_normGamma1, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_normBeta1, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_normGamma2, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_normBeta2, PersistentTensorRole.Biases);

    }

    private void InitializeParameters()
    {
        T half = NumOps.FromDouble(0.5);

        // Token shift mixing coefficients initialized to 0.5
        for (int i = 0; i < _modelDimension; i++)
        {
            _timeMixR[i] = half;
            _timeMixK[i] = half;
            _timeMixV[i] = half;
            _timeMixA[i] = half;
            _timeMixB[i] = half;
            _channelMixR[i] = half;
            _channelMixK[i] = half;
        }

        InitializeProjection(_receptanceWeights);
        InitializeProjection(_keyWeights);
        InitializeProjection(_valueWeights);
        // RWKV's reference initialization keeps the time-mix output projection
        // at zero so each newly initialized block begins as an exact residual
        // identity. A generic Xavier matrix here lets some otherwise-valid seeds
        // amplify the recurrent state on the very first optimizer step, producing
        // non-finite gradients before global clipping can act.
        _outputWeights.Fill(NumOps.Zero);

        // v7: State evolution projections - initialized for stable decay
        // Decay LoRA. w1 is ZERO so the data-dependent term starts at exactly tanh(0) = 0 and the
        // decay begins precisely at the w0 ramp below; w2 is orthogonal with gain 0.1. This is the
        // reference scheme, and the zero start is the load-bearing part — the previous full-rank
        // InitializeProjection(_aWeights) injected noise into the decay logit before a single step.
        _w1.Fill(NumOps.Zero);
        new OrthogonalInitializationStrategy<T>(0.1).InitializeWeights(_w2, _loraRank, _modelDimension);
        // Decay logit init, per the reference implementation (RWKV-LM RWKV-v7):
        //     www[n] = -6 + 6 * (n/(C-1))^(1 + ratio_0_to_1^0.3)
        //     w0[n]  = www[n] + 0.5 + zigzag[n] * 2.5
        // giving a per-channel RAMP of timescales rather than one shared value. Under the kernel's
        // w_t = exp(-e^(-1/2) * sigmoid(d_t)) this spans retention from about 0.55 (fast-forgetting
        // channels) to nearly 1.0 (near-permanent memory), which is the paper's (0.5453, 1) range.
        //
        // This replaces a flat Fill(-1.0), whose comment claimed "sigmoid(-1) ~ 0.27 retention". That
        // was true of the OLD kernel, which used sigmoid(d) directly as the retention; under Eq. 17
        // the same value gives ~0.85 on EVERY channel — uniform, long, and outside the spread the
        // architecture relies on to mix short- and long-range memory.
        //
        // ratio_0_to_1 is layer_id/(n_layer-1) in the reference. This block does not receive a layer
        // index, so the exponent uses ratio_0_to_1 = 0 (exponent 1, a linear ramp) — the reference's
        // first-layer schedule. Threading a layer index through would let later layers use the
        // steeper curve the reference gives them.
        for (int n = 0; n < _modelDimension; n++)
        {
            double frac = _modelDimension > 1 ? (double)n / (_modelDimension - 1) : 0.0;
            double zz = _headDimension > 1
                ? ((n % _headDimension) - (_headDimension - 1) / 2.0) / ((_headDimension - 1) / 2.0)
                : 0.0;
            double zigzag = zz * Math.Abs(zz);
            _aBias[n] = NumOps.FromDouble(-6.0 + 6.0 * frac + 0.5 + zigzag * 2.5);
        }

        // ICL-rate LoRA, same scheme. No tanh on this path in the reference — the sigmoid applied to
        // the sum bounds it.
        _a1.Fill(NumOps.Zero);
        new OrthogonalInitializationStrategy<T>(0.1).InitializeWeights(_a2, _loraRank, _modelDimension);
        _bBias.Fill(NumOps.FromDouble(0.0));

        // Removal- and injection-key scales, initialised as in the reference implementation
        // (RWKV-LM RWKV-v7): k_k = 0.71 - linear*0.1 with linear[n] = n/(C-1) - 0.5, and
        // k_a = 1.02. The k_k ramp gives early channels a slightly larger removal scale than late
        // ones; k_a just above 1 starts the injection key a touch beyond a plain delta rule.
        for (int n = 0; n < _modelDimension; n++)
        {
            double linear = _modelDimension > 1 ? (double)n / (_modelDimension - 1) - 0.5 : 0.0;
            _kk[n] = NumOps.FromDouble(0.71 - linear * 0.1);
        }
        _ka.Fill(NumOps.FromDouble(1.02));

        // r_k bonus scale, and the gate LoRA (same zero/orthogonal scheme as the decay and ICL pairs).
        _rk.Fill(NumOps.FromDouble(-0.04));

        // Value-residual gate. v0 = 0.73 - linear*0.4 puts sigmoid(v0) near 0.6-0.7 at init, so a fresh
        // deeper layer already leans toward the first layer.s values and learns its way off that.
        for (int n = 0; n < _modelDimension; n++)
        {
            double vlin = _modelDimension > 1 ? (double)n / (_modelDimension - 1) - 0.5 : 0.0;
            _v0[n] = NumOps.FromDouble(0.73 - vlin * 0.4);
        }
        _v1.Fill(NumOps.Zero);
        new OrthogonalInitializationStrategy<T>(0.1).InitializeWeights(_v2, _mvLoraRank, _modelDimension);
        _g1.Fill(NumOps.Zero);
        new OrthogonalInitializationStrategy<T>(0.1).InitializeWeights(_g2, _gateLoraRank, _modelDimension);
        // x_g uses the same 0.2 exponent as x_r in the reference's token-shift ramp.
        for (int n = 0; n < _modelDimension; n++)
        {
            double ddd = _modelDimension > 1 ? (double)n / _modelDimension : 0.0;
            _timeMixG[n] = NumOps.FromDouble(1.0 - Math.Pow(ddd, 0.2));
        }

        _groupNormGamma.Fill(NumOps.One);
        _groupNormBeta.Fill(NumOps.Zero);

        InitializeProjection(_channelKeyWeights);
        // The channel-mix value projection is likewise zero-initialized in the
        // reference RWKV recipe. The key path still receives gradients through
        // this projection after the first update while the residual stream starts
        // numerically stable.
        _channelValueWeights.Fill(NumOps.Zero);
        InitializeProjection(_channelReceptanceWeights);

        _normGamma1.Fill(NumOps.One);
        _normBeta1.Fill(NumOps.Zero);
        _normGamma2.Fill(NumOps.One);
        _normBeta2.Fill(NumOps.Zero);

        // Arena workspace: pre-allocate forward pass buffers for zero-allocation hot path
        Workspace = new LayerWorkspace<T>(timestepCount: 10, sequenceCount: 12);
        // TimeMixing timestep buffers
        Workspace.DeclareTimestep(TsRInput, _modelDimension);
        Workspace.DeclareTimestep(TsKInput, _modelDimension);
        Workspace.DeclareTimestep(TsVInput, _modelDimension);
        Workspace.DeclareTimestep(TsAInput, _modelDimension);
        Workspace.DeclareTimestep(TsBInput, _modelDimension);
        Workspace.DeclareTimestep(TsWkvOut, _modelDimension);
        // ChannelMixing timestep buffers
        Workspace.DeclareTimestep(TsCmRInput, _modelDimension);
        Workspace.DeclareTimestep(TsCmKInput, _modelDimension);
        Workspace.DeclareTimestep(TsCmKSiLU, _ffnDimension);
        // TimeMixing sequence buffers
        Workspace.DeclareSequence(SqAllR, _modelDimension);
        Workspace.DeclareSequence(SqAllK, _modelDimension);
        Workspace.DeclareSequence(SqAllV, _modelDimension);
        Workspace.DeclareSequence(SqAllA, _modelDimension);
        Workspace.DeclareSequence(SqAllB, _modelDimension);
        Workspace.DeclareSequence(SqAllWkv, _modelDimension);
        Workspace.DeclareSequence(SqAllWkvPre, _modelDimension);
        Workspace.DeclareSequence(SqAllWkvGated, _modelDimension);
        // ChannelMixing sequence buffers
        Workspace.DeclareSequence(SqCmAllRGate, _modelDimension);
        Workspace.DeclareSequence(SqCmAllVProj, _modelDimension);
        Workspace.DeclareSequence(SqCmAllSiLU, _ffnDimension);
        Workspace.DeclareSequence(SqCmAllKProj, _ffnDimension);
    }

    /// <summary>Gets the workspace, throwing if not initialized.</summary>
    private LayerWorkspace<T> Ws => Workspace
        ?? throw new InvalidOperationException("RWKV7Block workspace not initialized.");

    // Workspace buffer indices — TimeMixing timestep buffers
    private const int TsRInput = 0, TsKInput = 1, TsVInput = 2;
    private const int TsAInput = 3, TsBInput = 4, TsWkvOut = 5;
    // Workspace buffer indices — ChannelMixing timestep buffers
    private const int TsCmRInput = 7, TsCmKInput = 8, TsCmKSiLU = 9;
    // Workspace buffer indices — TimeMixing sequence buffers
    private const int SqAllR = 0, SqAllK = 1, SqAllV = 2, SqAllA = 3;
    private const int SqAllB = 4, SqAllWkv = 5, SqAllWkvPre = 6, SqAllWkvGated = 7;
    // Workspace buffer indices — ChannelMixing sequence buffers
    private const int SqCmAllRGate = 8, SqCmAllVProj = 9;
    // FFN-dimension sequence buffers (separate indices since different shape suffix)
    private const int SqCmAllSiLU = 10, SqCmAllKProj = 11;

    private void InitializeProjection(Tensor<T> tensor)
    {
        InitializeLayerWeights(tensor, tensor.Shape[0], tensor.Shape[1]);
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input) => ForwardWithValueResidual(input, null).Output;

    /// <summary>
    /// Forward pass that also threads the RWKV-7 value residual, mirroring the reference's
    /// <c>x, v_first = block(x, v_first)</c>.
    /// </summary>
    /// <param name="input">The block input.</param>
    /// <param name="vFirst">The first layer's value projection, or <c>null</c> when this IS the first layer.</param>
    /// <returns>The block output, and the v_first to hand to the next block.</returns>
    /// <remarks>
    /// v_first travels as an ordinary VALUE through the call chain rather than through shared mutable
    /// state. That is what keeps it a normal edge on the tape — the same reason PyTorch expresses it
    /// as a plain tuple return — and it removes any question of which block is "first" at clone or
    /// deserialize time: whoever is handed <c>null</c> is first.
    /// </remarks>
    internal (Tensor<T> Output, Tensor<T> VFirst) ForwardWithValueResidual(Tensor<T> input, Tensor<T>? vFirst)
    {
        _incomingVFirst = vFirst;
        _publishedVFirst = null;
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

        // Pre-size workspace for this forward pass
        Ws.BeginForward(batchSize, seqLen);

        // Time mixing sub-layer with residual
        var normed1 = ApplyLayerNorm(input3D, _normGamma1, _normBeta1, batchSize, seqLen);
        _lastNormed1 = normed1;
        var timeMixOut = TimeMixingForward(normed1, batchSize, seqLen);
        _lastTimeMixOutput = timeMixOut;
        var afterTimeMix = Engine.TensorAdd(input3D, timeMixOut);
        _lastAfterTimeMix = afterTimeMix;

        // Channel mixing sub-layer with residual
        var normed2 = ApplyLayerNorm(afterTimeMix, _normGamma2, _normBeta2, batchSize, seqLen);
        _lastNormed2 = normed2;
        var channelMixOut = ChannelMixingForward(normed2, batchSize, seqLen);
        _lastChannelMixOutput = channelMixOut;
        var output3D = Engine.TensorAdd(afterTimeMix, channelMixOut);

        var result = ApplyActivation(output3D);
        _lastOutput = result;

        if (rank == 2)
            return (Engine.Reshape(result, new[] { seqLen, _modelDimension }),
                    _publishedVFirst ?? vFirst ?? result);

        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
            outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen;
        outputShape[rank - 1] = _modelDimension;
        // _publishedVFirst is set by TimeMixingForward: this block's own value projection when it is
        // the first layer, otherwise the one it was handed, passed straight through.
        return (Engine.Reshape(result, outputShape), _publishedVFirst ?? vFirst ?? result);
    }

    /// <summary>
    /// Time mixing forward with WKV-7 dynamic state evolution kernel.
    /// </summary>
    private Tensor<T> TimeMixingForward(Tensor<T> x, int batchSize, int seqLen)
    {
        // Per-step outputs are concatenated on the autodiff tape (end of loop) rather
        // than written into a rented buffer via SafeSetSlice, which detached the output.
        var outputSlices = new System.Collections.Generic.List<Tensor<T>>(seqLen);

        // State: [batch, numHeads, headDim, headDim] - matrix-valued per head.
        // Statefulness is for autoregressive streaming inference only; in training each
        // sequence is independent, so start from a fresh zero state every Forward (otherwise
        // repeated forwards over the same input are not idempotent — carried state poisons
        // finite-difference gradient checks). Streaming callers run in inference mode.
        var state = (!IsTrainingMode && _recurrentState != null)
            ? _recurrentState
            : new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });
        var xPrev = (!IsTrainingMode && _prevToken != null)
            ? _prevToken
            : new Tensor<T>(new[] { batchSize, _modelDimension });

        // Cache intermediate values for backward — zero allocation via workspace
        var allR = Ws.Sequence(SqAllR);
        var allK = Ws.Sequence(SqAllK);
        var allV = Ws.Sequence(SqAllV);
        var allA = Ws.Sequence(SqAllA);
        var allB = Ws.Sequence(SqAllB);
        var allWkv = Ws.Sequence(SqAllWkv);
        var allWkvPreGate = Ws.Sequence(SqAllWkvPre);
        var allWkvGated = Ws.Sequence(SqAllWkvGated);

        // Token-shift mix coefficients as [1, modelDim] rows for broadcasting over
        // batch. Computed once (tape-connected to the mix vectors) and reused each
        // timestep. Expressing the lerp mix*x_t + (1-mix)*x_prev in Engine ops means
        // (a) every matmul input is a FRESH tensor — the prior code wrote each
        // timestep's r/k/v/a/b input into a single reused workspace buffer, so the
        // tape (which saves references, not snapshots) read only the LAST timestep's
        // values during backward and produced wrong projection-weight gradients — and
        // (b) the mix coefficients stay on the autodiff graph.
        // ---- #1464 throughput: token-shift + the r/k/v/a/b projections do NOT depend on the
        // recurrent WKV state, so they are computed for the WHOLE sequence in ONE batched GEMM each
        // (over [batch*seqLen, modelDim]) instead of seqLen separate per-timestep GEMMs. Only the
        // WKV state recurrence below stays sequential. Every op is still on the autodiff tape, so
        // the projection-weight gradients are identical to the per-step formulation — clone-parity
        // and training results are unchanged; this is purely a per-step-overhead reduction.
        var ones1D = Tensor<T>.CreateDefault(new[] { _modelDimension }, NumOps.One);
        // Mix coefficients as [1, 1, modelDim] so they broadcast over [batch, seqLen, modelDim].
        var mixR3 = Engine.Reshape(_timeMixR, new[] { 1, 1, _modelDimension });
        var mixK3 = Engine.Reshape(_timeMixK, new[] { 1, 1, _modelDimension });
        var mixV3 = Engine.Reshape(_timeMixV, new[] { 1, 1, _modelDimension });
        var mixA3 = Engine.Reshape(_timeMixA, new[] { 1, 1, _modelDimension });
        var mixB3 = Engine.Reshape(_timeMixB, new[] { 1, 1, _modelDimension });
        var mixG3 = Engine.Reshape(_timeMixG, new[] { 1, 1, _modelDimension });
        var invR3 = Engine.Reshape(Engine.TensorSubtract(ones1D, _timeMixR), new[] { 1, 1, _modelDimension });
        var invK3 = Engine.Reshape(Engine.TensorSubtract(ones1D, _timeMixK), new[] { 1, 1, _modelDimension });
        var invV3 = Engine.Reshape(Engine.TensorSubtract(ones1D, _timeMixV), new[] { 1, 1, _modelDimension });
        var invA3 = Engine.Reshape(Engine.TensorSubtract(ones1D, _timeMixA), new[] { 1, 1, _modelDimension });
        var invB3 = Engine.Reshape(Engine.TensorSubtract(ones1D, _timeMixB), new[] { 1, 1, _modelDimension });
        var invG3 = Engine.Reshape(Engine.TensorSubtract(ones1D, _timeMixG), new[] { 1, 1, _modelDimension });

        // Token-shifted input over the whole sequence: xShifted[:, t, :] = x[:, t-1, :], with the
        // t=0 slot taken from the previous token (zeros in training; the streaming cache otherwise).
        // GetSliceAlongDimension / TensorSliceAxis / TensorConcatenate record their backward on the
        // tape (AiDotNet.Tensors >= 0.85.3, PR #487), so the shift's gradient scatters to the right
        // positions exactly as the prior per-step zero-copy slice did.
        var xPrev0 = (!IsTrainingMode && _prevToken != null)
            ? Engine.Reshape(_prevToken, new[] { batchSize, 1, _modelDimension })
            : new Tensor<T>(new[] { batchSize, 1, _modelDimension });
        Tensor<T> xShifted = seqLen > 1
            ? Engine.TensorConcatenate(new[] { xPrev0, Engine.TensorNarrow(x, 1, 0, seqLen - 1) }, axis: 1)
            : xPrev0;

        // Batched token-shift lerps: mix*x_t + (1-mix)*x_prev over all timesteps at once.
        var rIn = Engine.TensorAdd(Engine.TensorMultiply(x, mixR3), Engine.TensorMultiply(xShifted, invR3));
        var kIn = Engine.TensorAdd(Engine.TensorMultiply(x, mixK3), Engine.TensorMultiply(xShifted, invK3));
        var vIn = Engine.TensorAdd(Engine.TensorMultiply(x, mixV3), Engine.TensorMultiply(xShifted, invV3));
        var aIn = Engine.TensorAdd(Engine.TensorMultiply(x, mixA3), Engine.TensorMultiply(xShifted, invA3));
        var bIn = Engine.TensorAdd(Engine.TensorMultiply(x, mixB3), Engine.TensorMultiply(xShifted, invB3));
        var gIn = Engine.TensorAdd(Engine.TensorMultiply(x, mixG3), Engine.TensorMultiply(xShifted, invG3));

        // Batched projections: [batch*seqLen, modelDim] @ [modelDim, modelDim] -> reshape back to 3D.
        int bsl = batchSize * seqLen;
        var Rall = Engine.Reshape(Engine.TensorMatMul(Engine.Reshape(rIn, new[] { bsl, _modelDimension }), _receptanceWeights), new[] { batchSize, seqLen, _modelDimension });
        var Kall = Engine.Reshape(Engine.TensorMatMul(Engine.Reshape(kIn, new[] { bsl, _modelDimension }), _keyWeights), new[] { batchSize, seqLen, _modelDimension });
        var Vall = Engine.Reshape(Engine.TensorMatMul(Engine.Reshape(vIn, new[] { bsl, _modelDimension }), _valueWeights), new[] { batchSize, seqLen, _modelDimension });
        // Value residual (arXiv:2503.14456). The first layer publishes its value projection; every
        // layer above blends toward it, per channel:
        //     v = v + (v_first - v) * sigmoid(v0 + (x_v @ v1) @ v2)
        // Built entirely from IEngine ops so the blend is a normal set of tape edges and the gradient
        // reaches v0/v1/v2 AND flows back down into the producing layer's value projection.
        if (_incomingVFirst is null)
        {
            _publishedVFirst = Vall;
        }
        else
        {
            var vGate = Engine.Sigmoid(Engine.TensorAdd(
                Engine.Reshape(
                    Engine.TensorMatMul(
                        Engine.TensorMatMul(Engine.Reshape(vIn, new[] { bsl, _modelDimension }), _v1),
                        _v2),
                    new[] { batchSize, seqLen, _modelDimension }),
                Engine.Reshape(_v0, new[] { 1, 1, _modelDimension })));
            Vall = Engine.TensorAdd(
                Vall,
                Engine.TensorMultiply(Engine.TensorSubtract(_incomingVFirst, Vall), vGate));
            _publishedVFirst = _incomingVFirst;
        }

        var aBias3 = Engine.Reshape(_aBias, new[] { 1, 1, _modelDimension });
        var bBias3 = Engine.Reshape(_bBias, new[] { 1, 1, _modelDimension });
        // Decay logit:  w = w0 + tanh(x_w @ w1) @ w2      (low-rank, tanh-bounded)
        // ICL logit:    a = a0 + (x_a @ a1) @ a2          (low-rank; sigmoid applied below)
        // Both per the reference. The tanh on the decay path is what keeps the logit from drifting
        // once channels are initialised near the top of the paper's (0.5453, 1) retention range.
        var Aall = Engine.TensorAdd(
            Engine.Reshape(
                Engine.TensorMatMul(
                    Engine.Tanh(Engine.TensorMatMul(Engine.Reshape(aIn, new[] { bsl, _modelDimension }), _w1)),
                    _w2),
                new[] { batchSize, seqLen, _modelDimension }),
            aBias3);
        var Ball = Engine.TensorAdd(
            Engine.Reshape(
                Engine.TensorMatMul(
                    Engine.TensorMatMul(Engine.Reshape(bIn, new[] { bsl, _modelDimension }), _a1),
                    _a2),
                new[] { batchSize, seqLen, _modelDimension }),
            bBias3);

        // ---- #1464: the entire WKV state recurrence (diagonal decay + rank-1 injection + gated
        // readout) runs in ONE fused, differentiable engine op instead of ~10 tape micro-ops per
        // timestep. The kernel applies the r/a/b sigmoids internally and records a single tape node
        // whose backward is the BPTT adjoint of the recurrence, so the projection-weight gradients
        // are identical to the per-step formulation (clone-parity preserved) — it just removes the
        // per-timestep tape-dispatch overhead that made the memorization test exceed the 180s budget.
        //   S_t[di,vi] = sigmoid(a)[di]*S_{t-1}[di,vi] + (sigmoid(b)[di]*k[di])*v[vi]
        //   wkv_t[di]  = sigmoid(r)[di] * sum_vi S_t[di,vi]*k[vi]
        // Generalised delta rule inputs (arXiv:2503.14456, Eq. 17). The kernel takes kappa
        // PRE-normalisation and forms -kappaHat and (a (*) kappaHat) itself, matching the reference
        // kernel call RWKV7_CLAMPW_CUDA(r, w, kTilde, v, -kk, kk*a); it likewise derives
        // w_t = exp(-e^(-1/2) * sigmoid(d_t)) from the decay LOGIT, so Aall is passed unactivated.
        //
        //   a_t      = sigmoid(Ball)              in-context learning rate, in (0,1)
        //   kappa_t  = k_t (*) k_k                removal key, L2-normalised per head in-kernel
        //   kTilde_t = k_t (*) (1 + (a_t-1)(*)k_a) value-injection key
        //
        // a_t is materialised here rather than inside the kernel because kTilde needs it too.
        var kk3 = Engine.Reshape(_kk, new[] { 1, 1, _modelDimension });
        var ka3 = Engine.Reshape(_ka, new[] { 1, 1, _modelDimension });
        var iclRate = Engine.Sigmoid(Ball);
        var kappa = Engine.TensorMultiply(Kall, kk3);
        var kTilde = Engine.TensorMultiply(
            Kall,
            Engine.TensorAddScalar(
                Engine.TensorMultiply(Engine.TensorSubtractScalar(iclRate, NumOps.One), ka3),
                NumOps.One));

        // Apply the paper's Global ICLR Multiplier c to the TRANSITION's removal term only. Scaling
        // the in-context learning rate here is exactly equivalent to
        //   A_t = diag(w_t) - c * kappaHat^T(a (*) kappaHat)
        // because the kernel forms the removal as kappaHat^T(a (*) kappaHat). It must NOT be applied
        // to the replacement key: Eq. 7 defines kTilde = k (*) lerp(1, a, alpha) with no c, so the
        // kTilde above deliberately keeps the unscaled iclRate.
        var iclRateTransition = _globalIclrMultiplier == 1.0
            ? iclRate
            : Engine.TensorMultiplyScalar(iclRate, NumOps.FromDouble(_globalIclrMultiplier));

        var wkvAll = Engine.Rwkv7SequenceForward(Rall, kappa, kTilde, Vall, Aall, iclRateTransition, _numHeads);

        // Group-normalize (per head, per position) and project to the output — both batched over all
        // positions as [batch*seqLen, modelDim], so NO per-timestep ops remain in time-mixing.
        var wkv2d = Engine.Reshape(wkvAll, new[] { bsl, _modelDimension });
        var normed2d = ApplyGroupNorm(wkv2d, bsl);

        // r_k bonus, AFTER the group norm and before the gate, per the reference:
        //     x = x + ((r (*) k (*) r_k).sum(dim=-1, keepdim=True) * v)
        // A direct current-token route that bypasses the recurrent state: the head's own r-k
        // agreement, scaled per channel by r_k, admits a copy of v straight to the output. Reduced
        // over headDim and broadcast back across it.
        var rHeads = Engine.Reshape(Rall, new[] { bsl, _numHeads, _headDimension });
        var kHeads = Engine.Reshape(Kall, new[] { bsl, _numHeads, _headDimension });
        var vHeads = Engine.Reshape(Vall, new[] { bsl, _numHeads, _headDimension });
        var rk3 = Engine.Reshape(_rk, new[] { 1, _numHeads, _headDimension });
        var rkAgreement = Engine.ReduceSum(
            Engine.TensorMultiply(Engine.TensorMultiply(rHeads, kHeads), rk3),
            new[] { 2 }, keepDims: true);                       // [bsl, numHeads, 1]
        var bonus2d = Engine.Reshape(
            Engine.TensorMultiply(vHeads, rkAgreement),          // broadcasts over headDim
            new[] { bsl, _modelDimension });
        normed2d = Engine.TensorAdd(normed2d, bonus2d);

        // Output gate: g = sigmoid(x_g @ g1) @ g2, applied multiplicatively before the projection.
        // g1 is zero-initialised, so sigmoid(0) = 0.5 and g starts as a constant 0.5 @ g2 rather
        // than at zero — the gate is live from step one, unlike the decay/ICL LoRAs.
        var gate2d = Engine.TensorMatMul(
            Engine.Sigmoid(Engine.TensorMatMul(Engine.Reshape(gIn, new[] { bsl, _modelDimension }), _g1)),
            _g2);
        normed2d = Engine.TensorMultiply(normed2d, gate2d);
        // These projections are paper-faithfully initialized to exactly zero, but
        // they are trainable parameters. Keep an explicit graph dependency on the
        // parameter so compilation cannot mistake the initial value for a static
        // sparse inference constant and discard its backward path.
        var trainableOutputWeights = Engine.Reshape(
            _outputWeights, _outputWeights.Shape.ToArray());
        var output = Engine.Reshape(
            Engine.TensorMatMul(normed2d, trainableOutputWeights),
            new[] { batchSize, seqLen, _modelDimension });

        // Recurrent-state persistence. In TRAINING each sequence is independent and the carried state is
        // never read back (the read gate above is `!IsTrainingMode`), so we skip the recurrence entirely —
        // keeping the #1464 per-step-overhead win. In INFERENCE the autoregressive streaming contract
        // requires the final WKV state S_T so the next call can continue the sequence; the fused
        // Rwkv7SequenceForward returns only the gated outputs (not S_T), so compute S_T from the same
        // documented recurrence (off-tape; inference only):
        //   S_t[di,vi] = sigmoid(A_t)[di]*S_{t-1}[di,vi] + (sigmoid(B_t)[di]*K_t[di])*V_t[vi]
        // seeded from the prior state (`state`) so token-by-token streaming accumulates correctly.
        if (IsTrainingMode)
        {
            // Training sequences are independent — clear ALL carried state,
            // including the token-shift caches. Leaving _prevToken /
            // _prevChannelToken live would mix the first inference token with
            // the last training token if the block is reused for streaming
            // without an explicit ResetState().
            _recurrentState = null;
            _prevToken = null;
            _prevChannelToken = null;
        }
        else
        {
            _recurrentState = ComputeFinalWkvState(state, Aall, Ball, Kall, Vall, batchSize, seqLen);
            _prevToken = seqLen > 0 ? x.GetSliceAlongDimension(seqLen - 1, 1) : xPrev;
        }

        // Cache for backward
        _cachedWkvOut = allWkv;
        _cachedR = allR;
        _cachedK = allK;
        _cachedV = allV;
        _cachedTimeMixNormed1 = x;
        _cachedWkvGated = allWkvGated;

        // Previous tokens reconstructed from _cachedTimeMixNormed1 during backward:
        // prevToken[t=0] = zeros, prevToken[t>0] = x[t-1]

        return output;
    }

    /// <summary>
    /// Computes the final WKV recurrent state S_T after processing a sequence, for autoregressive streaming
    /// inference. Mirrors the recurrence the fused <c>Rwkv7SequenceForward</c> kernel applies internally:
    /// <c>S_t[h,di,vi] = sigmoid(A_t)[h,di]*S_{t-1}[h,di,vi] + sigmoid(B_t)[h,di]*K_t[h,di]*V_t[h,vi]</c>,
    /// per head. Runs off the autodiff tape (scalar arithmetic over the projected A/B/K/V values) since the
    /// streaming state carries no gradient — it only seeds the next inference call.
    /// </summary>
    /// <param name="seed">The prior state to continue from (zeros on the first call), [batch, heads, headDim, headDim].</param>
    /// <param name="aAll">Decay projection A over the sequence, [batch, seqLen, modelDim].</param>
    /// <param name="bAll">Injection-gate projection B over the sequence, [batch, seqLen, modelDim].</param>
    /// <param name="kAll">Key projection over the sequence, [batch, seqLen, modelDim].</param>
    /// <param name="vAll">Value projection over the sequence, [batch, seqLen, modelDim].</param>
    /// <param name="batchSize">The batch size.</param>
    /// <param name="seqLen">The sequence length.</param>
    /// <returns>The final state S_T, shape [batch, numHeads, headDim, headDim].</returns>
    private Tensor<T> ComputeFinalWkvState(
        Tensor<T> seed, Tensor<T> aAll, Tensor<T> bAll, Tensor<T> kAll, Tensor<T> vAll, int batchSize, int seqLen)
    {
        int hd = _headDimension;
        var s = new Tensor<T>(new[] { batchSize, _numHeads, hd, hd });

        // Seed from the prior state so streaming (seqLen == 1 per call) accumulates across calls.
        for (int bi = 0; bi < batchSize; bi++)
            for (int h = 0; h < _numHeads; h++)
                for (int di = 0; di < hd; di++)
                    for (int vi = 0; vi < hd; vi++)
                        s[new[] { bi, h, di, vi }] = seed[new[] { bi, h, di, vi }];

        for (int t = 0; t < seqLen; t++)
            for (int bi = 0; bi < batchSize; bi++)
                for (int h = 0; h < _numHeads; h++)
                    for (int di = 0; di < hd; di++)
                    {
                        int idx = (h * hd) + di;
                        T a = Sigmoid(aAll[new[] { bi, t, idx }]);
                        T bk = NumOps.Multiply(Sigmoid(bAll[new[] { bi, t, idx }]), kAll[new[] { bi, t, idx }]);
                        for (int vi = 0; vi < hd; vi++)
                        {
                            T v = vAll[new[] { bi, t, (h * hd) + vi }];
                            T prev = s[new[] { bi, h, di, vi }];
                            s[new[] { bi, h, di, vi }] = NumOps.Add(NumOps.Multiply(a, prev), NumOps.Multiply(bk, v));
                        }
                    }

        return s;
    }

    private T Sigmoid(T x) =>
        NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, NumOps.FromDouble(Math.Exp(-NumOps.ToDouble(x)))));

    /// <summary>
    /// Channel mixing forward with SiLU gating (RWKV-7 style).
    /// </summary>
    private Tensor<T> ChannelMixingForward(Tensor<T> x, int batchSize, int seqLen)
    {
        // ---- #1464: channel mixing is purely position-wise (token-shift + a SiLU-gated FFN, NO
        // recurrence), so the whole sub-layer runs as batched GEMMs over [batch*seqLen, modelDim]
        // — no per-timestep loop. The previous per-step loop issued ~8 Engine dispatches × seqLen ×
        // numLayers (≈16K dispatches/forward at seqLen=512, 4 layers); that per-op DISPATCH overhead
        // — NOT GEMM FLOPs (the GEMMs run at 30–90 GFLOP/s) — dominated the forward (~9.5s measured
        // for one Predict) and the training step. The tape backs the gradients automatically (the
        // layer has no manual backward), so weights/activations are identical to the per-step form.
        int bsl = batchSize * seqLen;
        var ones1D = Tensor<T>.CreateDefault(new[] { _modelDimension }, NumOps.One);
        var mixR3 = Engine.Reshape(_channelMixR, new[] { 1, 1, _modelDimension });
        var mixK3 = Engine.Reshape(_channelMixK, new[] { 1, 1, _modelDimension });
        var invR3 = Engine.Reshape(Engine.TensorSubtract(ones1D, _channelMixR), new[] { 1, 1, _modelDimension });
        var invK3 = Engine.Reshape(Engine.TensorSubtract(ones1D, _channelMixK), new[] { 1, 1, _modelDimension });

        // Token-shift over the whole sequence: xShifted[:, t, :] = x[:, t-1, :].
        var xPrev0 = (!IsTrainingMode && _prevChannelToken != null)
            ? Engine.Reshape(_prevChannelToken, new[] { batchSize, 1, _modelDimension })
            : new Tensor<T>(new[] { batchSize, 1, _modelDimension });
        Tensor<T> xShifted = seqLen > 1
            ? Engine.TensorConcatenate(new[] { xPrev0, Engine.TensorNarrow(x, 1, 0, seqLen - 1) }, axis: 1)
            : xPrev0;

        var rIn = Engine.TensorAdd(Engine.TensorMultiply(x, mixR3), Engine.TensorMultiply(xShifted, invR3));
        var kIn = Engine.TensorAdd(Engine.TensorMultiply(x, mixK3), Engine.TensorMultiply(xShifted, invK3));

        // r = sigmoid(W_r · rIn); k = W_k · kIn; SiLU(k); v = W_v · SiLU(k); out = sigmoid(r) · v.
        var rGate = Engine.Sigmoid(Engine.TensorMatMul(Engine.Reshape(rIn, new[] { bsl, _modelDimension }), _channelReceptanceWeights)); // [bsl, modelDim]
        var kProj = Engine.TensorMatMul(Engine.Reshape(kIn, new[] { bsl, _modelDimension }), _channelKeyWeights); // [bsl, ffnDim]
        var kSiLU = Engine.TensorMultiply(kProj, Engine.Sigmoid(kProj));
        // Preserve the trainable dependency for the other zero-initialized RWKV
        // projection for the same reason as the time-mix output projection above.
        var trainableChannelValueWeights = Engine.Reshape(
            _channelValueWeights, _channelValueWeights.Shape.ToArray());
        var vProj = Engine.TensorMatMul(kSiLU, trainableChannelValueWeights); // [bsl, modelDim]
        var y = Engine.TensorMultiply(rGate, vProj); // [bsl, modelDim]

        // Carry the channel token-shift cache only for the inference streaming
        // contract; in training the sequences are independent and a live cache
        // would leak the last training token into a later inference call.
        if (!IsTrainingMode && seqLen > 0)
        {
            _prevChannelToken = x.GetSliceAlongDimension(seqLen - 1, 1);
        }

        return Engine.Reshape(y, new[] { batchSize, seqLen, _modelDimension });
    }

    /// <summary>
    /// Applies group normalization across heads. Each head's dimensions are normalized independently.
    /// </summary>
    private Tensor<T> ApplyGroupNorm(Tensor<T> input, int batchSize)
    {
        // Group normalization per head: each head's _headDimension contiguous
        // channels share a (mean, variance), with independent gamma/beta per
        // channel. That's exactly Engine.GroupNorm with numGroups=numHeads over
        // a [batchSize, modelDimension, 1, 1] 4D reshape. The previous manual
        // loop did 4 × batchSize × numHeads × headDimension scalar NumOps calls
        // (mean + variance + normalize + scale/bias passes); this is one fused
        // call.
        int modelDim = _numHeads * _headDimension;
        var input4D = Engine.Reshape(input, new[] { batchSize, modelDim, 1, 1 });
        // eps = 64e-5, per the reference (nn.GroupNorm(H, C, eps=64e-5)) — NOT the 1e-6 that was here.
        // The WKV readout can leave a head with near-zero variance, and 1/sqrt(var + 1e-6) then
        // amplifies that head enormously; 64e-5 is ~640x larger and deliberately damps it. This is a
        // stability choice in the architecture, not a rounding detail.
        var output4D = Engine.GroupNorm(input4D, _numHeads, _groupNormGamma, _groupNormBeta, 64e-5, out _, out _);
        return Engine.Reshape(output4D, input._shape);
    }

    /// <summary>
    /// Applies layer normalization.
    /// </summary>
    private Tensor<T> ApplyLayerNorm(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta,
        int batchSize, int seqLen)
    {
        var shaped = Engine.Reshape(input, new[] { batchSize, seqLen, _modelDimension });
        return Engine.LayerNorm(shaped, gamma, beta, 1e-6, out _, out _);
    }


    /// <summary>
    /// GroupNorm backward pass. Computes input gradient with proper normalization chain rule.
    /// Uses the standard formula: dx = (1/N) * gamma / std * (N*dy - sum(dy) - xhat*sum(dy*xhat))
    /// </summary>
    private Tensor<T> GroupNormBackward(Tensor<T> dOutput, Tensor<T> input, int batchSize)
    {
        var dInput = TensorAllocator.Rent<T>(input._shape);
        T eps = NumOps.FromDouble(1e-6);

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                // Compute mean and variance for this head
                T mean = NumOps.Zero;
                for (int d = 0; d < _headDimension; d++)
                    mean = NumOps.Add(mean, input[new[] { bi, dimStart + d }]);
                mean = NumOps.Divide(mean, NumOps.FromDouble(_headDimension));

                T variance = NumOps.Zero;
                for (int d = 0; d < _headDimension; d++)
                {
                    T diff = NumOps.Subtract(input[new[] { bi, dimStart + d }], mean);
                    variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
                }
                variance = NumOps.Divide(variance, NumOps.FromDouble(_headDimension));
                T stdInv = NumOps.Divide(NumOps.One, NumOps.Sqrt(NumOps.Add(variance, eps)));

                // Compute xhat and the two sums needed for BN backward
                T sumDy = NumOps.Zero;
                T sumDyXhat = NumOps.Zero;
                var xhat = new T[_headDimension];
                var dyGamma = new T[_headDimension];

                for (int d = 0; d < _headDimension; d++)
                {
                    int flatD = dimStart + d;
                    xhat[d] = NumOps.Multiply(NumOps.Subtract(input[new[] { bi, flatD }], mean), stdInv);
                    dyGamma[d] = NumOps.Multiply(dOutput[new[] { bi, flatD }], _groupNormGamma[flatD]);
                    sumDy = NumOps.Add(sumDy, dyGamma[d]);
                    sumDyXhat = NumOps.Add(sumDyXhat, NumOps.Multiply(dyGamma[d], xhat[d]));

                    // Accumulate groupNorm gamma/beta gradients
                    if (_groupNormGammaGrad is not null)
                        _groupNormGammaGrad[flatD] = NumOps.Add(_groupNormGammaGrad[flatD],
                            NumOps.Multiply(dOutput[new[] { bi, flatD }], xhat[d]));
                    if (_groupNormBetaGrad is not null)
                        _groupNormBetaGrad[flatD] = NumOps.Add(_groupNormBetaGrad[flatD],
                            dOutput[new[] { bi, flatD }]);
                }

                // Standard normalization backward:
                // dx = (1/N) * stdInv * (N * dyGamma - sumDy - xhat * sumDyXhat)
                T invN = NumOps.Divide(NumOps.One, NumOps.FromDouble(_headDimension));
                for (int d = 0; d < _headDimension; d++)
                {
                    int flatD = dimStart + d;
                    T nDy = NumOps.Multiply(NumOps.FromDouble(_headDimension), dyGamma[d]);
                    T term = NumOps.Subtract(NumOps.Subtract(nDy, sumDy),
                        NumOps.Multiply(xhat[d], sumDyXhat));
                    dInput[new[] { bi, flatD }] = NumOps.Multiply(NumOps.Multiply(invN, stdInv), term);
                }
            }
        }

        return dInput;
    }

    /// <summary>
    /// Accumulates gradients for LayerNorm gamma and beta parameters.
    /// LayerNorm: output = gamma * (input - mean) / std + beta
    /// </summary>
    /// <summary>
    /// Copies a 2D slice [batch, dim] into a 3D tensor [batch, seqLen, dim] at time position t.
    /// Uses explicit per-element copy to avoid SetSlice position bugs.
    /// </summary>
    private static void SafeSetSlice(Tensor<T> dest, int t, Tensor<T> slice, int batch, int dim)
    {
        for (int bi = 0; bi < batch; bi++)
            for (int d = 0; d < dim; d++)
                dest[new[] { bi, t, d }] = slice[new[] { bi, d }];
    }


    /// <summary>
    /// LayerNorm backward: returns gradient w.r.t. input.
    /// Forward: output = gamma * (x - mean) / std + beta
    /// Backward: dx = (gamma*dy - normalized*mean(gamma*dy*normalized)) / std (per the LN paper)
    /// </summary>
    private Tensor<T> LayerNormBackward(Tensor<T> dOutput, Tensor<T> input,
        Tensor<T> gamma, int batchSize, int seqLen)
    {
        var dInput = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _modelDimension });
        T eps = NumOps.FromDouble(1e-6);
        T invDim = NumOps.FromDouble(1.0 / _modelDimension);

        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                T mean = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++)
                    mean = NumOps.Add(mean, input[new[] { b, t, d }]);
                mean = NumOps.Multiply(mean, invDim);

                T variance = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++)
                {
                    T diff = NumOps.Subtract(input[new[] { b, t, d }], mean);
                    variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
                }
                variance = NumOps.Multiply(variance, invDim);
                T stdInv = NumOps.Divide(NumOps.One, NumOps.Sqrt(NumOps.Add(variance, eps)));

                // Compute normalized and the correction term
                var normalized = new T[_modelDimension];
                T dotProduct = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++)
                {
                    normalized[d] = NumOps.Multiply(NumOps.Subtract(input[new[] { b, t, d }], mean), stdInv);
                    dotProduct = NumOps.Add(dotProduct,
                        NumOps.Multiply(NumOps.Multiply(gamma[d], dOutput[new[] { b, t, d }]), normalized[d]));
                }
                T meanDot = NumOps.Multiply(dotProduct, invDim);

                // Compute sum of gamma * dy for the mean subtraction correction
                T sumGammaDy = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++)
                    sumGammaDy = NumOps.Add(sumGammaDy, NumOps.Multiply(gamma[d], dOutput[new[] { b, t, d }]));
                T meanGammaDy = NumOps.Multiply(sumGammaDy, invDim);

                for (int d = 0; d < _modelDimension; d++)
                {
                    T gammaDy = NumOps.Multiply(gamma[d], dOutput[new[] { b, t, d }]);
                    T correction = NumOps.Multiply(normalized[d], meanDot);
                    dInput[new[] { b, t, d }] = NumOps.Multiply(
                        NumOps.Subtract(NumOps.Subtract(gammaDy, meanGammaDy), correction), stdInv);
                }
            }
        }

        return dInput;
    }

    private void AccumulateLayerNormGradients(Tensor<T> dOutput, Tensor<T> input,
        Tensor<T> gamma, ref Tensor<T>? gammaGrad, ref Tensor<T>? betaGrad,
        int batchSize, int seqLen)
    {
        if (gammaGrad == null || betaGrad == null) return;

        T eps = NumOps.FromDouble(1e-6);
        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                // Compute mean and std for this position
                T mean = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++)
                    mean = NumOps.Add(mean, input[new[] { b, t, d }]);
                mean = NumOps.Divide(mean, NumOps.FromDouble(_modelDimension));

                T variance = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++)
                {
                    T diff = NumOps.Subtract(input[new[] { b, t, d }], mean);
                    variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
                }
                variance = NumOps.Divide(variance, NumOps.FromDouble(_modelDimension));
                T stdInv = NumOps.Divide(NumOps.One, NumOps.Sqrt(NumOps.Add(variance, eps)));

                for (int d = 0; d < _modelDimension; d++)
                {
                    T normalized = NumOps.Multiply(
                        NumOps.Subtract(input[new[] { b, t, d }], mean), stdInv);
                    T dOut = dOutput[new[] { b, t, d }];

                    gammaGrad[d] = NumOps.Add(gammaGrad[d], NumOps.Multiply(dOut, normalized));
                    betaGrad[d] = NumOps.Add(betaGrad[d], dOut);
                }
            }
        }
    }

    private void InitializeGradients()
    {
        _timeMixRGrad = new Tensor<T>([_modelDimension]);
        _timeMixKGrad = new Tensor<T>([_modelDimension]);
        _timeMixVGrad = new Tensor<T>([_modelDimension]);
        _timeMixAGrad = new Tensor<T>([_modelDimension]);
        _timeMixBGrad = new Tensor<T>([_modelDimension]);
        _receptanceWeightsGrad = new Tensor<T>([_modelDimension, _modelDimension]);
        _keyWeightsGrad = new Tensor<T>([_modelDimension, _modelDimension]);
        _valueWeightsGrad = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputWeightsGrad = new Tensor<T>([_modelDimension, _modelDimension]);
        _w1Grad = new Tensor<T>([_modelDimension, _loraRank]);
        _w2Grad = new Tensor<T>([_loraRank, _modelDimension]);
        _aBiasGrad = new Tensor<T>([_modelDimension]);
        _a1Grad = new Tensor<T>([_modelDimension, _loraRank]);
        _a2Grad = new Tensor<T>([_loraRank, _modelDimension]);
        _bBiasGrad = new Tensor<T>([_modelDimension]);
        _v0Grad = new Tensor<T>([_modelDimension]);
        _v1Grad = new Tensor<T>([_modelDimension, _mvLoraRank]);
        _v2Grad = new Tensor<T>([_mvLoraRank, _modelDimension]);
        _rkGrad = new Tensor<T>([_numHeads, _headDimension]);
        _timeMixGGrad = new Tensor<T>([_modelDimension]);
        _g1Grad = new Tensor<T>([_modelDimension, _gateLoraRank]);
        _g2Grad = new Tensor<T>([_gateLoraRank, _modelDimension]);
        _kkGrad = new Tensor<T>([_modelDimension]);
        _kaGrad = new Tensor<T>([_modelDimension]);
        _groupNormGammaGrad = new Tensor<T>([_modelDimension]);
        _groupNormBetaGrad = new Tensor<T>([_modelDimension]);
        _channelMixRGrad = new Tensor<T>([_modelDimension]);
        _channelMixKGrad = new Tensor<T>([_modelDimension]);
        _channelKeyWeightsGrad = new Tensor<T>([_modelDimension, _ffnDimension]);
        _channelValueWeightsGrad = new Tensor<T>([_ffnDimension, _modelDimension]);
        _channelReceptanceWeightsGrad = new Tensor<T>([_modelDimension, _modelDimension]);
        _normGamma1Grad = new Tensor<T>([_modelDimension]);
        _normBeta1Grad = new Tensor<T>([_modelDimension]);
        _normGamma2Grad = new Tensor<T>([_modelDimension]);
        _normBeta2Grad = new Tensor<T>([_modelDimension]);
    }

    /// <summary>
    /// Accumulates gradients for the output projection weights from the time mixing sub-layer.
    /// </summary>
    private void AccumulateTimeMixGradients(Tensor<T> dOutput, Tensor<T> normedInput,
        int batchSize, int seqLen)
    {
        if (_cachedWkvOut == null || _outputWeightsGrad == null) return;

        // dOutput flows through output projection: y = wkv * W_out
        // dW_out += wkv^T * dOutput
        for (int t = 0; t < seqLen; t++)
        {
            var dOut_t = dOutput.GetSliceAlongDimension(t, 1);  // [batch, modelDim]
            var wkv_t = _cachedWkvOut.GetSliceAlongDimension(t, 1);

            var dW = Engine.TensorMatMul(wkv_t.Transpose(new[] { 1, 0 }), dOut_t);

            for (int i = 0; i < _modelDimension; i++)
                for (int j = 0; j < _modelDimension; j++)
                    _outputWeightsGrad[new[] { i, j }] = NumOps.Add(
                        _outputWeightsGrad[new[] { i, j }], dW[new[] { i, j }]);
        }
    }

    /// <summary>
    /// Accumulates gradients for the channel mixing sub-layer.
    /// </summary>
    /// <remarks>
    /// Channel mixing: output = sigmoid(r) * (W_v * SiLU(W_k * kInput))
    /// Gradients computed for W_v (value weights) and W_r (receptance weights).
    /// </remarks>
    private void AccumulateChannelMixGradients(Tensor<T> dOutput, Tensor<T> normedInput,
        int batchSize, int seqLen)
    {
        if (_cachedChannelRGate == null || _cachedChannelSiLU == null ||
            _cachedChannelVProj == null || _channelValueWeightsGrad == null ||
            _channelReceptanceWeightsGrad == null)
            return;

        for (int t = 0; t < seqLen; t++)
        {
            var dOut_t = dOutput.GetSliceAlongDimension(t, 1);  // [batch, modelDim]
            var rGate_t = _cachedChannelRGate.GetSliceAlongDimension(t, 1);  // [batch, modelDim]
            var siLU_t = _cachedChannelSiLU.GetSliceAlongDimension(t, 1);    // [batch, ffnDim]
            var vProj_t = _cachedChannelVProj.GetSliceAlongDimension(t, 1);  // [batch, modelDim]

            // dOutput flows through: y = rGate * vProj
            // d(vProj) = rGate * dOutput (element-wise)
            var dVProj = Engine.TensorMultiply(rGate_t, dOut_t);

            // d(W_v) += SiLU^T * d(vProj)
            var dWv = Engine.TensorMatMul(siLU_t.Transpose(new[] { 1, 0 }), dVProj);
            for (int i = 0; i < _ffnDimension; i++)
                for (int j = 0; j < _modelDimension; j++)
                    _channelValueWeightsGrad[new[] { i, j }] = NumOps.Add(
                        _channelValueWeightsGrad[new[] { i, j }], dWv[new[] { i, j }]);

            // d(rGate) = vProj * dOutput (element-wise) - for receptance weight gradients
            // rGate = sigmoid(W_r * rInput), so d(W_r) involves sigmoid derivative
            // Simplified: accumulate W_r gradient via dRGate * sigmoid'(rGate) * rInput^T
            var dRGate = Engine.TensorMultiply(vProj_t, dOut_t);

            // sigmoid derivative: sigmoid(x) * (1 - sigmoid(x)) = rGate * (1 - rGate)
            var sigmoidDeriv = new Tensor<T>(rGate_t._shape);
            for (int bi = 0; bi < batchSize; bi++)
                for (int d = 0; d < _modelDimension; d++)
                {
                    T r = rGate_t[new[] { bi, d }];
                    sigmoidDeriv[new[] { bi, d }] = NumOps.Multiply(r, NumOps.Subtract(NumOps.One, r));
                }

            var dRProj = Engine.TensorMultiply(dRGate, sigmoidDeriv);
            var normed_t = normedInput.GetSliceAlongDimension(t, 1);

            var dWr = Engine.TensorMatMul(normed_t.Transpose(new[] { 1, 0 }), dRProj);
            for (int i = 0; i < _modelDimension; i++)
                for (int j = 0; j < _modelDimension; j++)
                    _channelReceptanceWeightsGrad[new[] { i, j }] = NumOps.Add(
                        _channelReceptanceWeightsGrad[new[] { i, j }], dWr[new[] { i, j }]);
        }
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_timeMixRGrad == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);

        var allParams = GetAllParameterTensors();
        var allGrads = GetAllGradientTensors();

        for (int i = 0; i < allParams.Length; i++)
        {
            var grad = allGrads[i];
            if (grad != null)
            {
                var updated = Engine.TensorAdd(allParams[i],
                    Engine.TensorMultiplyScalar(grad, negLR));
                CopyTensorData(updated, allParams[i]);
            }
        }
    }

    private static void CopyTensorData(Tensor<T> source, Tensor<T> destination)
    {
        for (int i = 0; i < source.Length; i++)
            destination[i] = source[i];
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var parameters = new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));
        int index = 0;
        foreach (var tensor in GetAllParameterTensors())
        {
            for (int i = 0; i < tensor.Length; i++)
                parameters[index++] = tensor[i];
        }
        return parameters;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameterGradients()
    {
        var allParams = GetAllParameterTensors();
        var allGrads = GetAllGradientTensors();
        var result = new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));
        int index = 0;

        for (int i = 0; i < allParams.Length; i++)
        {
            var grad = allGrads[i];
            if (grad != null)
            {
                for (int j = 0; j < grad.Length; j++)
                    result[index++] = grad[j];
            }
            else
            {
                index += allParams[i].Length;
            }
        }

        return result;
    }

    /// <inheritdoc />
    public override void ClearGradients()
    {
        base.ClearGradients();
        _timeMixRGrad = null;
        _timeMixKGrad = null;
        _timeMixVGrad = null;
        _timeMixAGrad = null;
        _timeMixBGrad = null;
        _receptanceWeightsGrad = null;
        _keyWeightsGrad = null;
        _valueWeightsGrad = null;
        _outputWeightsGrad = null;
        _w1Grad = null;
        _w2Grad = null;
        _aBiasGrad = null;
        _a1Grad = null;
        _a2Grad = null;
        _bBiasGrad = null;
        _v0Grad = null;
        _v1Grad = null;
        _v2Grad = null;
        _rkGrad = null;
        _timeMixGGrad = null;
        _g1Grad = null;
        _g2Grad = null;
        _kkGrad = null;
        _kaGrad = null;
        _groupNormGammaGrad = null;
        _groupNormBetaGrad = null;
        _channelMixRGrad = null;
        _channelMixKGrad = null;
        _channelKeyWeightsGrad = null;
        _channelValueWeightsGrad = null;
        _channelReceptanceWeightsGrad = null;
        _normGamma1Grad = null;
        _normBeta1Grad = null;
        _normGamma2Grad = null;
        _normBeta2Grad = null;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}");

        int index = 0;
        foreach (var tensor in GetAllParameterTensors())
        {
            for (int i = 0; i < tensor.Length; i++)
                tensor[i] = parameters[index++];
        }
    }

    /// <summary>
    /// Gets the recurrent state for autoregressive continuation.
    /// </summary>
    public Tensor<T>? GetRecurrentState() => _recurrentState?.Clone();

    /// <summary>
    /// Sets the recurrent state for autoregressive continuation.
    /// </summary>
    public void SetRecurrentState(Tensor<T>? state)
    {
        _recurrentState = state?.Clone();
    }

    /// <summary>
    /// Gets the previous token for token-shift continuation.
    /// </summary>
    public Tensor<T>? GetPreviousToken() => _prevToken?.Clone();

    /// <summary>
    /// Sets the previous token for token-shift continuation.
    /// </summary>
    public void SetPreviousToken(Tensor<T>? token)
    {
        _prevToken = token?.Clone();
        _prevChannelToken = null;  // Channel prev token resets with time mixing prev token
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastTimeMixOutput = null;
        _lastChannelMixOutput = null;
        _lastNormed1 = null;
        _lastNormed2 = null;
        _lastAfterTimeMix = null;
        _originalInputShape = null;
        _cachedWkvOut = null;
        _cachedChannelRGate = null;
        _cachedChannelSiLU = null;
        _cachedChannelVProj = null;
        _recurrentState = null;
        _prevToken = null;
        _prevChannelToken = null;
        ClearAllGradients();
    }

    private void ClearAllGradients()
    {
        _timeMixRGrad = null;
        _timeMixKGrad = null;
        _timeMixVGrad = null;
        _timeMixAGrad = null;
        _timeMixBGrad = null;
        _receptanceWeightsGrad = null;
        _keyWeightsGrad = null;
        _valueWeightsGrad = null;
        _outputWeightsGrad = null;
        _w1Grad = null;
        _w2Grad = null;
        _aBiasGrad = null;
        _a1Grad = null;
        _a2Grad = null;
        _bBiasGrad = null;
        _v0Grad = null;
        _v1Grad = null;
        _v2Grad = null;
        _rkGrad = null;
        _timeMixGGrad = null;
        _g1Grad = null;
        _g2Grad = null;
        _kkGrad = null;
        _kaGrad = null;
        _groupNormGammaGrad = null;
        _groupNormBetaGrad = null;
        _channelMixRGrad = null;
        _channelMixKGrad = null;
        _channelKeyWeightsGrad = null;
        _channelValueWeightsGrad = null;
        _channelReceptanceWeightsGrad = null;
        _normGamma1Grad = null;
        _normBeta1Grad = null;
        _normGamma2Grad = null;
        _normBeta2Grad = null;
    }

    /// <summary>All trainable tensors, for gradient diagnostics in tests.</summary>
    internal Tensor<T>[] ParameterTensorsForDiagnostics => GetAllParameterTensors();

    /// <summary>Names positionally matching <see cref="ParameterTensorsForDiagnostics"/>.</summary>
    internal static string[] ParameterNamesForDiagnostics =>
    [
        "timeMixR", "timeMixK", "timeMixV", "timeMixA", "timeMixB",
        "receptanceWeights", "keyWeights", "valueWeights", "outputWeights",
        "w1", "w2", "aBias", "a1", "a2", "bBias", "kk", "ka", "rk", "timeMixG", "g1", "g2", "v0", "v1", "v2",
        "groupNormGamma", "groupNormBeta",
        "channelMixR", "channelMixK",
        "channelKeyWeights", "channelValueWeights", "channelReceptanceWeights",
        "normGamma1", "normBeta1", "normGamma2", "normBeta2"
    ];

    private Tensor<T>[] GetAllParameterTensors() =>
    [
        _timeMixR, _timeMixK, _timeMixV, _timeMixA, _timeMixB,
        _receptanceWeights, _keyWeights, _valueWeights, _outputWeights,
        _w1, _w2, _aBias, _a1, _a2, _bBias, _kk, _ka, _rk, _timeMixG, _g1, _g2, _v0, _v1, _v2,
        _groupNormGamma, _groupNormBeta,
        _channelMixR, _channelMixK,
        _channelKeyWeights, _channelValueWeights, _channelReceptanceWeights,
        _normGamma1, _normBeta1, _normGamma2, _normBeta2
    ];

    private Tensor<T>?[] GetAllGradientTensors() =>
    [
        _timeMixRGrad, _timeMixKGrad, _timeMixVGrad, _timeMixAGrad, _timeMixBGrad,
        _receptanceWeightsGrad, _keyWeightsGrad, _valueWeightsGrad, _outputWeightsGrad,
        _w1Grad, _w2Grad, _aBiasGrad, _a1Grad, _a2Grad, _bBiasGrad, _kkGrad, _kaGrad, _rkGrad, _timeMixGGrad, _g1Grad, _g2Grad, _v0Grad, _v1Grad, _v2Grad,
        _groupNormGammaGrad, _groupNormBetaGrad,
        _channelMixRGrad, _channelMixKGrad,
        _channelKeyWeightsGrad, _channelValueWeightsGrad, _channelReceptanceWeightsGrad,
        _normGamma1Grad, _normBeta1Grad, _normGamma2Grad, _normBeta2Grad
    ];

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["NumHeads"] = _numHeads.ToString();
        metadata["HeadDimension"] = _headDimension.ToString();
        metadata["FFNDimension"] = _ffnDimension.ToString();
        metadata["Architecture"] = "RWKV-7";
        return metadata;
    }
}

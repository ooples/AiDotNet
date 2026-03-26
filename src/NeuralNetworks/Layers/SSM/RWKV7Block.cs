using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

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
public class RWKV7Block<T> : LayerBase<T>
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
    private Tensor<T> _receptanceWeights;
    private Tensor<T> _keyWeights;
    private Tensor<T> _valueWeights;
    private Tensor<T> _outputWeights;

    // v7: Dynamic state evolution projections
    private Tensor<T> _aWeights;  // [modelDim, modelDim] projects to per-head diagonal decay
    private Tensor<T> _aBias;     // [modelDim]
    private Tensor<T> _bWeights;  // [modelDim, modelDim] projects to per-head additive injection
    private Tensor<T> _bBias;     // [modelDim]

    // v7: Group norm on WKV output (per head)
    private Tensor<T> _groupNormGamma;  // [modelDim]
    private Tensor<T> _groupNormBeta;   // [modelDim]

    // ============ Channel Mixing Parameters ============

    private Tensor<T> _channelMixR;  // [modelDim]
    private Tensor<T> _channelMixK;  // [modelDim]
    private Tensor<T> _channelKeyWeights;        // [modelDim, ffnDim]
    private Tensor<T> _channelValueWeights;      // [ffnDim, modelDim]
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
    private Tensor<T>? _cachedChannelKProj;   // [batch, seqLen, ffnDim] W_k * kInput (pre-SiLU)

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
    private Tensor<T>? _aWeightsGrad;
    private Tensor<T>? _aBiasGrad;
    private Tensor<T>? _bWeightsGrad;
    private Tensor<T>? _bBiasGrad;
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

    /// <inheritdoc />
    public override int ParameterCount
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

        _aWeights = new Tensor<T>([modelDimension, modelDimension]);
        _aBias = new Tensor<T>([modelDimension]);
        _bWeights = new Tensor<T>([modelDimension, modelDimension]);
        _bBias = new Tensor<T>([modelDimension]);

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
        InitializeProjection(_outputWeights);

        // v7: State evolution projections - initialized for stable decay
        InitializeProjection(_aWeights);
        _aBias.Fill(NumOps.FromDouble(-1.0));  // Initial log-decay: sigmoid(-1) ~ 0.27 retention per step

        InitializeProjection(_bWeights);
        _bBias.Fill(NumOps.FromDouble(0.0));

        _groupNormGamma.Fill(NumOps.One);
        _groupNormBeta.Fill(NumOps.Zero);

        InitializeProjection(_channelKeyWeights);
        InitializeProjection(_channelValueWeights);
        InitializeProjection(_channelReceptanceWeights);

        _normGamma1.Fill(NumOps.One);
        _normBeta1.Fill(NumOps.Zero);
        _normGamma2.Fill(NumOps.One);
        _normBeta2.Fill(NumOps.Zero);
    }

    private void InitializeProjection(Tensor<T> tensor)
    {
        InitializeLayerWeights(tensor, tensor.Shape[0], tensor.Shape[1]);
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape.ToArray();

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
            return result.Reshape(seqLen, _modelDimension);

        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
            outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen;
        outputShape[rank - 1] = _modelDimension;
        return result.Reshape(outputShape);
    }

    /// <summary>
    /// Time mixing forward with WKV-7 dynamic state evolution kernel.
    /// </summary>
    private Tensor<T> TimeMixingForward(Tensor<T> x, int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        // State: [batch, numHeads, headDim, headDim] - matrix-valued per head
        var state = _recurrentState ?? new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });
        var xPrev = _prevToken ?? new Tensor<T>(new[] { batchSize, _modelDimension });

        // Cache intermediate values for backward
        var allR = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var allK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var allV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var allA = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var allB = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var allWkv = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var allWkvPreGate = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var allWkvGated = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        for (int t = 0; t < seqLen; t++)
        {
            var x_t = x.GetSliceAlongDimension(t, 1);  // [batch, modelDim]

            // Token shift: lerp between current and previous token
            var rInput = new Tensor<T>(new[] { batchSize, _modelDimension });
            var kInput = new Tensor<T>(new[] { batchSize, _modelDimension });
            var vInput = new Tensor<T>(new[] { batchSize, _modelDimension });
            var aInput = new Tensor<T>(new[] { batchSize, _modelDimension });
            var bInput = new Tensor<T>(new[] { batchSize, _modelDimension });

            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _modelDimension; d++)
                {
                    T curr = x_t[new[] { bi, d }];
                    T prev = xPrev[new[] { bi, d }];

                    rInput[new[] { bi, d }] = NumOps.Add(
                        NumOps.Multiply(_timeMixR[d], curr),
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, _timeMixR[d]), prev));
                    kInput[new[] { bi, d }] = NumOps.Add(
                        NumOps.Multiply(_timeMixK[d], curr),
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, _timeMixK[d]), prev));
                    vInput[new[] { bi, d }] = NumOps.Add(
                        NumOps.Multiply(_timeMixV[d], curr),
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, _timeMixV[d]), prev));
                    aInput[new[] { bi, d }] = NumOps.Add(
                        NumOps.Multiply(_timeMixA[d], curr),
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, _timeMixA[d]), prev));
                    bInput[new[] { bi, d }] = NumOps.Add(
                        NumOps.Multiply(_timeMixB[d], curr),
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, _timeMixB[d]), prev));
                }
            }

            // Project to r, k, v, a, b
            var r = Engine.TensorMatMul(rInput, _receptanceWeights);
            var k = Engine.TensorMatMul(kInput, _keyWeights);
            var v = Engine.TensorMatMul(vInput, _valueWeights);

            // v7: Dynamic state evolution parameters
            var aProj = Engine.TensorMatMul(aInput, _aWeights);
            var aBias2D = _aBias.Reshape(1, _modelDimension);
            aProj = Engine.TensorBroadcastAdd(aProj, aBias2D);

            var bProj = Engine.TensorMatMul(bInput, _bWeights);
            var bBias2D = _bBias.Reshape(1, _modelDimension);
            bProj = Engine.TensorBroadcastAdd(bProj, bBias2D);

            // Cache for backward — use explicit copy to avoid SetSlice position bugs
            SafeSetSlice(allR, t, r, batchSize, _modelDimension);
            SafeSetSlice(allK, t, k, batchSize, _modelDimension);
            SafeSetSlice(allV, t, v, batchSize, _modelDimension);
            SafeSetSlice(allA, t, aProj, batchSize, _modelDimension);
            SafeSetSlice(allB, t, bProj, batchSize, _modelDimension);

            // WKV-7 kernel per head
            var wkvOutput = new Tensor<T>(new[] { batchSize, _modelDimension });

            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int hi = 0; hi < _numHeads; hi++)
                {
                    int dimStart = hi * _headDimension;

                    // Compute a_t (sigmoid for stable decay in [0,1]) and b_t per head dimension
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatD = dimStart + di;

                        // a_t: sigmoid gives retention factor in [0, 1]
                        double aRaw = NumOps.ToDouble(aProj[new[] { bi, flatD }]);
                        double aVal = 1.0 / (1.0 + Math.Exp(-aRaw));  // sigmoid

                        // b_t: used for additive injection scaling
                        double bVal = NumOps.ToDouble(bProj[new[] { bi, flatD }]);
                        double bSigmoid = 1.0 / (1.0 + Math.Exp(-bVal));  // sigmoid for stability

                        T aFactor = NumOps.FromDouble(aVal);
                        T bFactor = NumOps.FromDouble(bSigmoid);

                        // State update: S[di, :] = a_t[di] * S[di, :] + b_t[di] * k_t[di] * v_t[:]
                        // kVal is k[di] used for the rank-1 outer product update
                        T kVal = k[new[] { bi, flatD }];

                        T wkvNum = NumOps.Zero;
                        for (int vi = 0; vi < _headDimension; vi++)
                        {
                            int flatV = dimStart + vi;
                            T vVal = v[new[] { bi, flatV }];

                            T prevState = state[new[] { bi, hi, di, vi }];

                            // WKV-7 state update: S = diag(a) * S + b * outer(k, v)
                            T newState = NumOps.Add(
                                NumOps.Multiply(aFactor, prevState),
                                NumOps.Multiply(bFactor, NumOps.Multiply(kVal, vVal)));

                            state[new[] { bi, hi, di, vi }] = newState;

                            // Read from state: output[di] = sum_vi(S[di, vi] * k[dimStart + vi])
                            // Note: kCol uses flatV (= dimStart + vi), NOT kVal (= k[di])
                            T kCol = k[new[] { bi, flatV }];
                            wkvNum = NumOps.Add(wkvNum, NumOps.Multiply(newState, kCol));
                        }

                        // Apply receptance gate: sigmoid(r) * wkv
                        double rRaw = NumOps.ToDouble(r[new[] { bi, flatD }]);
                        double rGate = 1.0 / (1.0 + Math.Exp(-rRaw));

                        wkvOutput[new[] { bi, flatD }] = NumOps.Multiply(
                            NumOps.FromDouble(rGate), wkvNum);
                    }
                }
            }

            // Cache for backward
            SafeSetSlice(allWkvGated, t, wkvOutput, batchSize, _modelDimension);

            // Group normalization on WKV output (per head)
            var normedWkv = ApplyGroupNorm(wkvOutput, batchSize);
            SafeSetSlice(allWkv, t, normedWkv, batchSize, _modelDimension);

            // Output projection
            var y_t = Engine.TensorMatMul(normedWkv, _outputWeights);
            SafeSetSlice(output, t, y_t, batchSize, _modelDimension);

            xPrev = x_t;
        }

        // Store state for autoregressive inference
        _recurrentState = state;
        _prevToken = xPrev;

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
    /// Channel mixing forward with SiLU gating (RWKV-7 style).
    /// </summary>
    private Tensor<T> ChannelMixingForward(Tensor<T> x, int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var xPrev = _prevChannelToken ?? new Tensor<T>(new[] { batchSize, _modelDimension });

        // Caches for backward pass
        var allRGate = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var allSiLU = new Tensor<T>(new[] { batchSize, seqLen, _ffnDimension });
        var allVProj = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var allKProj = new Tensor<T>(new[] { batchSize, seqLen, _ffnDimension });

        for (int t = 0; t < seqLen; t++)
        {
            var x_t = x.GetSliceAlongDimension(t, 1);

            // Token shift
            var rInput = new Tensor<T>(new[] { batchSize, _modelDimension });
            var kInput = new Tensor<T>(new[] { batchSize, _modelDimension });

            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _modelDimension; d++)
                {
                    T curr = x_t[new[] { bi, d }];
                    T prev = xPrev[new[] { bi, d }];
                    rInput[new[] { bi, d }] = NumOps.Add(
                        NumOps.Multiply(_channelMixR[d], curr),
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, _channelMixR[d]), prev));
                    kInput[new[] { bi, d }] = NumOps.Add(
                        NumOps.Multiply(_channelMixK[d], curr),
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, _channelMixK[d]), prev));
                }
            }

            // r = sigmoid(W_r * rInput)
            var rProj = Engine.TensorMatMul(rInput, _channelReceptanceWeights);
            var rGate = Engine.Sigmoid(rProj);

            // k = W_k * kInput, then SiLU activation
            var kProj = Engine.TensorMatMul(kInput, _channelKeyWeights);  // [batch, ffnDim]

            // SiLU: x * sigmoid(x)
            var kSiLU = new Tensor<T>(new[] { batchSize, _ffnDimension });
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _ffnDimension; d++)
                {
                    double val = NumOps.ToDouble(kProj[new[] { bi, d }]);
                    double sigmoid = 1.0 / (1.0 + Math.Exp(-val));
                    kSiLU[new[] { bi, d }] = NumOps.FromDouble(val * sigmoid);
                }
            }

            // v = W_v * SiLU(k)
            var vProj = Engine.TensorMatMul(kSiLU, _channelValueWeights);  // [batch, modelDim]

            // output = sigmoid(r) * v
            var y_t = Engine.TensorMultiply(rGate, vProj);
            SafeSetSlice(output, t, y_t, batchSize, _modelDimension);

            // Cache for backward
            SafeSetSlice(allRGate, t, rGate, batchSize, _modelDimension);
            SafeSetSliceDim(allSiLU, t, kSiLU, batchSize, _ffnDimension);
            SafeSetSlice(allVProj, t, vProj, batchSize, _modelDimension);
            SafeSetSliceDim(allKProj, t, kProj, batchSize, _ffnDimension);

            xPrev = x_t;
        }

        _cachedChannelRGate = allRGate;
        _cachedChannelSiLU = allSiLU;
        _cachedChannelVProj = allVProj;
        _cachedChannelKProj = allKProj;
        _prevChannelToken = xPrev;

        return output;
    }

    /// <summary>
    /// Applies group normalization across heads. Each head's dimensions are normalized independently.
    /// </summary>
    private Tensor<T> ApplyGroupNorm(Tensor<T> input, int batchSize)
    {
        var output = new Tensor<T>(input.Shape.ToArray());
        T eps = NumOps.FromDouble(1e-6);

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                // Compute mean for this head's dimensions
                T mean = NumOps.Zero;
                for (int d = 0; d < _headDimension; d++)
                    mean = NumOps.Add(mean, input[new[] { bi, dimStart + d }]);
                mean = NumOps.Divide(mean, NumOps.FromDouble(_headDimension));

                // Compute variance
                T variance = NumOps.Zero;
                for (int d = 0; d < _headDimension; d++)
                {
                    T diff = NumOps.Subtract(input[new[] { bi, dimStart + d }], mean);
                    variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
                }
                variance = NumOps.Divide(variance, NumOps.FromDouble(_headDimension));
                T stdInv = NumOps.Divide(NumOps.One, NumOps.Sqrt(NumOps.Add(variance, eps)));

                // Normalize and apply scale/bias
                for (int d = 0; d < _headDimension; d++)
                {
                    int flatD = dimStart + d;
                    T normalized = NumOps.Multiply(
                        NumOps.Subtract(input[new[] { bi, flatD }], mean), stdInv);
                    output[new[] { bi, flatD }] = NumOps.Add(
                        NumOps.Multiply(_groupNormGamma[flatD], normalized),
                        _groupNormBeta[flatD]);
                }
            }
        }

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

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        grad3D = ApplyActivationDerivativeFromOutput(_lastOutput, grad3D);

        InitializeGradients();

        // === Channel mixing backward ===
        // Forward: output = afterTimeMix + channelMixOut
        //          channelMixOut = ChannelMixing(LayerNorm2(afterTimeMix))
        // Both paths (residual + channelMix) get the full upstream gradient
        var dChannelMixOut = grad3D;
        var dAfterTimeMix = grad3D; // residual path contribution

        if (_lastNormed2 != null && _lastChannelMixOutput != null && _lastAfterTimeMix != null)
        {
            // Accumulate channel mixing weight gradients
            AccumulateChannelMixGradients(dChannelMixOut, _lastNormed2, batchSize, seqLen);

            // Compute gradient flowing back THROUGH channel mixing to normed2 input
            // output_t = rGate * vProj, where rGate = sigmoid(W_r @ rInput), vProj = W_v @ SiLU(W_k @ kInput)
            // rInput = channelMixR * x + (1-channelMixR) * xPrev
            // kInput = channelMixK * x + (1-channelMixK) * xPrev
            var dNormed2 = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
            if (_cachedChannelRGate is null || _cachedChannelSiLU is null || _cachedChannelVProj is null || _cachedChannelKProj is null)
            {
                dNormed2 = dChannelMixOut; // fallback
            }
            else
            for (int t = 0; t < seqLen; t++)
            {
                var dOut_t = dChannelMixOut.GetSliceAlongDimension(t, 1).Clone();
                var rGate_t = _cachedChannelRGate.GetSliceAlongDimension(t, 1).Clone();
                var siLU_t = _cachedChannelSiLU.GetSliceAlongDimension(t, 1).Clone();
                var vProj_t = _cachedChannelVProj.GetSliceAlongDimension(t, 1).Clone();

                // dRGate = dOut * vProj, dVProj = dOut * rGate
                var dVProj = Engine.TensorMultiply(dOut_t, rGate_t);
                var dRGate = Engine.TensorMultiply(dOut_t, vProj_t);

                // dKSiLU = dVProj @ W_v^T
                var dKSiLU = Engine.TensorMatMul(dVProj, _channelValueWeights.Transpose(new[] { 1, 0 }));

                // dRProj = dRGate * sigmoid'(rProj) = dRGate * rGate * (1-rGate)
                var ones = Tensor<T>.CreateDefault(rGate_t.Shape.ToArray(), NumOps.One);
                var sigDeriv = Engine.TensorMultiply(rGate_t, Engine.TensorSubtract(ones, rGate_t));
                var dRProj = Engine.TensorMultiply(dRGate, sigDeriv);

                // dRInput = dRProj @ W_r^T → dX_from_r = dRInput * channelMixR
                var dRInput = Engine.TensorMatMul(dRProj, _channelReceptanceWeights.Transpose(new[] { 1, 0 }));

                // dKProj = dKSiLU * SiLU'(kProj) where SiLU'(x) = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
                var kProj_t = _cachedChannelKProj.GetSliceAlongDimension(t, 1).Clone();
                var dKProj = new Tensor<T>(new[] { batchSize, _ffnDimension });
                for (int bi = 0; bi < batchSize; bi++)
                {
                    for (int fd = 0; fd < _ffnDimension; fd++)
                    {
                        double x = NumOps.ToDouble(kProj_t[new[] { bi, fd }]);
                        double sig = 1.0 / (1.0 + Math.Exp(-x));
                        double siluDeriv = sig + x * sig * (1.0 - sig);
                        dKProj[new[] { bi, fd }] = NumOps.Multiply(
                            dKSiLU[new[] { bi, fd }], NumOps.FromDouble(siluDeriv));
                    }
                }
                // dKInput = dKProj @ W_k^T
                var dKInput = Engine.TensorMatMul(dKProj, _channelKeyWeights.Transpose(new[] { 1, 0 }));

                // dNormed2[t] = dRInput * channelMixR + dKInput * channelMixK
                for (int bi = 0; bi < batchSize; bi++)
                {
                    for (int d = 0; d < _modelDimension; d++)
                    {
                        T fromR = NumOps.Multiply(dRInput[new[] { bi, d }], _channelMixR[d]);
                        T fromK = NumOps.Multiply(dKInput[new[] { bi, d }], _channelMixK[d]);
                        dNormed2[new[] { bi, t, d }] = NumOps.Add(fromR, fromK);
                    }
                }
            }

            // LayerNorm2 backward: dAfterTimeMix += LN2.backward(dNormed2)
            // This is the MISSING gradient path that was causing 5-40% errors
            var dAfterTimeMixFromLN2 = LayerNormBackward(dNormed2, _lastAfterTimeMix,
                _normGamma2, batchSize, seqLen);

            // Accumulate LN2 gamma/beta gradients
            AccumulateLayerNormGradients(dChannelMixOut, _lastAfterTimeMix, _normGamma2,
                ref _normGamma2Grad, ref _normBeta2Grad, batchSize, seqLen);

            // Total gradient for afterTimeMix = residual + LN2 path
            dAfterTimeMix = Engine.TensorAdd(dAfterTimeMix, dAfterTimeMixFromLN2);
        }

        // === Time mixing backward ===
        // Forward: afterTimeMix = input + timeMixOut
        var dTimeMixOut = dAfterTimeMix;
        var dInput = dAfterTimeMix; // residual path

        if (_lastNormed1 != null && _cachedWkvOut != null && _cachedR != null)
        {
            AccumulateTimeMixGradients(dTimeMixOut, _lastNormed1, batchSize, seqLen);

            // Compute all time mixing parameter gradients
            AccumulateTimeMixParameterGradients(dTimeMixOut, _lastNormed1, batchSize, seqLen);
        }

        // Accumulate LN1 gamma/beta gradients
        AccumulateLayerNormGradients(dTimeMixOut, _lastInput, _normGamma1,
            ref _normGamma1Grad, ref _normBeta1Grad, batchSize, seqLen);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return dInput.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return dInput.Reshape(_originalInputShape);

        return dInput;
    }

    /// <summary>
    /// Computes gradients for all time mixing parameters: timeMixR/K/V/A/B,
    /// receptance/key/value weights, a/b weights+biases, groupNorm gamma/beta.
    /// </summary>
    private void AccumulateTimeMixParameterGradients(Tensor<T> dTimeMixOut, Tensor<T> normedInput,
        int batchSize, int seqLen)
    {
        if (_cachedWkvOut == null || _cachedR == null || _cachedK == null || _cachedV == null ||
            _cachedWkvGated == null)
            return;

        // Per-timestep backward through output projection → groupNorm → receptance gate
        for (int t = 0; t < seqLen; t++)
        {
            var dOut_t = dTimeMixOut.GetSliceAlongDimension(t, 1); // [batch, modelDim]
            var r_t = _cachedR.GetSliceAlongDimension(t, 1); // [batch, modelDim]
            var x_t = normedInput.GetSliceAlongDimension(t, 1); // [batch, modelDim]
            var gated_t = _cachedWkvGated.GetSliceAlongDimension(t, 1); // [batch, modelDim] pre-groupNorm

            // 1. Output projection backward: dNormedWkv = dOut @ W_out^T
            var dNormedWkv = Engine.TensorMatMul(dOut_t, _outputWeights.Transpose(new[] { 1, 0 }));

            // 2. GroupNorm backward (per-head normalization)
            // GroupNorm: for each head h, output[d] = gamma[d] * (x[d] - mean_h) / std_h + beta[d]
            // Backward: proper group norm backward with mean/std dependency
            var dGated = GroupNormBackward(dNormedWkv, gated_t, batchSize);

            // 3. Receptance gate backward
            // Forward: gated[d] = sigmoid(r[d]) * wkv_num[d]
            // d(gated)/d(r) = sigmoid'(r) * wkv_num = sig*(1-sig) * wkv_num
            // d(gated)/d(wkv_num) = sigmoid(r)
            // We need wkv_num = gated / sigmoid(r)

            var sigR = new Tensor<T>(new[] { batchSize, _modelDimension });
            var wkvNum = new Tensor<T>(new[] { batchSize, _modelDimension });
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _modelDimension; d++)
                {
                    T rVal = r_t[new[] { bi, d }];
                    T sig = NumOps.Divide(NumOps.One,
                        NumOps.Add(NumOps.One, NumOps.FromDouble(Math.Exp(-NumOps.ToDouble(rVal)))));
                    sigR[new[] { bi, d }] = sig;

                    // Recover wkv_num from gated output: wkv_num = gated / sigmoid(r)
                    T g = gated_t[new[] { bi, d }];
                    wkvNum[new[] { bi, d }] = NumOps.GreaterThan(NumOps.Abs(sig), NumOps.FromDouble(1e-10))
                        ? NumOps.Divide(g, sig) : NumOps.Zero;
                }
            }

            // dR = dGated * wkv_num * sigmoid'(r)
            var ones = Tensor<T>.CreateDefault(sigR.Shape.ToArray(), NumOps.One);
            var sigDeriv = Engine.TensorMultiply(sigR, Engine.TensorSubtract(ones, sigR));
            var dR = Engine.TensorMultiply(Engine.TensorMultiply(dGated, wkvNum), sigDeriv);

            // 4. dRInput = dR @ W_r^T
            var dRInput = Engine.TensorMatMul(dR, _receptanceWeights.Transpose(new[] { 1, 0 }));

            // 5. Token shift gradient: rInput = timeMixR * x_t + (1-timeMixR) * x_prev
            // dTimeMixR[d] += sum_b(dRInput[b,d] * (x_t[b,d] - x_prev[b,d]))
            Tensor<T> x_prev;
            if (t == 0)
            {
                x_prev = new Tensor<T>(new[] { batchSize, _modelDimension }); // zeros
            }
            else
            {
                x_prev = normedInput.GetSliceAlongDimension(t - 1, 1);
            }

            for (int d = 0; d < _modelDimension; d++)
            {
                T gradSum = NumOps.Zero;
                for (int bi = 0; bi < batchSize; bi++)
                {
                    T diff = NumOps.Subtract(x_t[new[] { bi, d }], x_prev[new[] { bi, d }]);
                    gradSum = NumOps.Add(gradSum, NumOps.Multiply(dRInput[new[] { bi, d }], diff));
                }
                if (_timeMixRGrad is not null)
                    _timeMixRGrad[d] = NumOps.Add(_timeMixRGrad[d], gradSum);
            }

            // 6. Receptance weight gradient: dW_r += rInput_t^T @ dR_t
            // Reconstruct rInput from token shift
            var rInput_t = new Tensor<T>(new[] { batchSize, _modelDimension });
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _modelDimension; d++)
                {
                    T curr = x_t[new[] { bi, d }];
                    T prev = x_prev[new[] { bi, d }];
                    rInput_t[new[] { bi, d }] = NumOps.Add(
                        NumOps.Multiply(_timeMixR[d], curr),
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, _timeMixR[d]), prev));
                }
            }
            if (_receptanceWeightsGrad is not null)
            {
                var dW = Engine.TensorMatMul(rInput_t.Transpose(new[] { 1, 0 }), dR);
                _receptanceWeightsGrad = Engine.TensorAdd(_receptanceWeightsGrad, dW);
            }

            // Similarly accumulate K, V, A, B token shift and weight gradients
            // (same pattern as R but through different paths — approximate contribution)
            var dK = Engine.TensorMatMul(dGated, _keyWeights.Transpose(new[] { 1, 0 }));
            var kInput_t = new Tensor<T>(new[] { batchSize, _modelDimension });
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _modelDimension; d++)
                {
                    T curr = x_t[new[] { bi, d }];
                    T prev = x_prev[new[] { bi, d }];
                    kInput_t[new[] { bi, d }] = NumOps.Add(
                        NumOps.Multiply(_timeMixK[d], curr),
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, _timeMixK[d]), prev));

                    T diff = NumOps.Subtract(curr, prev);
                    if (_timeMixKGrad is not null)
                        _timeMixKGrad[d] = NumOps.Add(_timeMixKGrad[d],
                            NumOps.Multiply(dK[new[] { bi, d }], diff));
                }
            }
            if (_keyWeightsGrad is not null)
            {
                var dWk = Engine.TensorMatMul(kInput_t.Transpose(new[] { 1, 0 }), dK);
                _keyWeightsGrad = Engine.TensorAdd(_keyWeightsGrad, dWk);
            }

            // V gradient
            var dV = Engine.TensorMatMul(dGated, _valueWeights.Transpose(new[] { 1, 0 }));
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _modelDimension; d++)
                {
                    T diff = NumOps.Subtract(x_t[new[] { bi, d }], x_prev[new[] { bi, d }]);
                    if (_timeMixVGrad is not null)
                        _timeMixVGrad[d] = NumOps.Add(_timeMixVGrad[d],
                            NumOps.Multiply(dV[new[] { bi, d }], diff));
                }
            }
            var vInput_t = new Tensor<T>(new[] { batchSize, _modelDimension });
            for (int bi = 0; bi < batchSize; bi++)
                for (int d = 0; d < _modelDimension; d++)
                    vInput_t[new[] { bi, d }] = NumOps.Add(
                        NumOps.Multiply(_timeMixV[d], x_t[new[] { bi, d }]),
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, _timeMixV[d]), x_prev[new[] { bi, d }]));
            if (_valueWeightsGrad is not null)
                _valueWeightsGrad = Engine.TensorAdd(_valueWeightsGrad,
                    Engine.TensorMatMul(vInput_t.Transpose(new[] { 1, 0 }), dV));

            // A and B gradients (through sigmoid activation)
            var dAInput = Engine.TensorMatMul(dGated, _aWeights.Transpose(new[] { 1, 0 }));
            for (int bi = 0; bi < batchSize; bi++)
                for (int d = 0; d < _modelDimension; d++)
                {
                    T diff = NumOps.Subtract(x_t[new[] { bi, d }], x_prev[new[] { bi, d }]);
                    if (_timeMixAGrad is not null)
                        _timeMixAGrad[d] = NumOps.Add(_timeMixAGrad[d],
                            NumOps.Multiply(dAInput[new[] { bi, d }], diff));
                }
            var aInput_t = new Tensor<T>(new[] { batchSize, _modelDimension });
            for (int bi = 0; bi < batchSize; bi++)
                for (int d = 0; d < _modelDimension; d++)
                    aInput_t[new[] { bi, d }] = NumOps.Add(
                        NumOps.Multiply(_timeMixA[d], x_t[new[] { bi, d }]),
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, _timeMixA[d]), x_prev[new[] { bi, d }]));
            if (_aWeightsGrad is not null)
                _aWeightsGrad = Engine.TensorAdd(_aWeightsGrad,
                    Engine.TensorMatMul(aInput_t.Transpose(new[] { 1, 0 }), dAInput));
            if (_aBiasGrad is not null)
                _aBiasGrad = Engine.TensorAdd(_aBiasGrad, Engine.ReduceSum(dAInput, new[] { 0 }));

            var dBInput = Engine.TensorMatMul(dGated, _bWeights.Transpose(new[] { 1, 0 }));
            for (int bi = 0; bi < batchSize; bi++)
                for (int d = 0; d < _modelDimension; d++)
                {
                    T diff = NumOps.Subtract(x_t[new[] { bi, d }], x_prev[new[] { bi, d }]);
                    if (_timeMixBGrad is not null)
                        _timeMixBGrad[d] = NumOps.Add(_timeMixBGrad[d],
                            NumOps.Multiply(dBInput[new[] { bi, d }], diff));
                }
            var bInput_t = new Tensor<T>(new[] { batchSize, _modelDimension });
            for (int bi = 0; bi < batchSize; bi++)
                for (int d = 0; d < _modelDimension; d++)
                    bInput_t[new[] { bi, d }] = NumOps.Add(
                        NumOps.Multiply(_timeMixB[d], x_t[new[] { bi, d }]),
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, _timeMixB[d]), x_prev[new[] { bi, d }]));
            if (_bWeightsGrad is not null)
                _bWeightsGrad = Engine.TensorAdd(_bWeightsGrad,
                    Engine.TensorMatMul(bInput_t.Transpose(new[] { 1, 0 }), dBInput));
            if (_bBiasGrad is not null)
                _bBiasGrad = Engine.TensorAdd(_bBiasGrad, Engine.ReduceSum(dBInput, new[] { 0 }));
        }
    }

    /// <summary>
    /// GroupNorm backward pass. Computes input gradient with proper normalization chain rule.
    /// Uses the standard formula: dx = (1/N) * gamma / std * (N*dy - sum(dy) - xhat*sum(dy*xhat))
    /// </summary>
    private Tensor<T> GroupNormBackward(Tensor<T> dOutput, Tensor<T> input, int batchSize)
    {
        var dInput = new Tensor<T>(input.Shape.ToArray());
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

    private static void SafeSetSliceDim(Tensor<T> dest, int t, Tensor<T> slice, int batch, int dim)
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
        var dInput = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
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
        _aWeightsGrad = new Tensor<T>([_modelDimension, _modelDimension]);
        _aBiasGrad = new Tensor<T>([_modelDimension]);
        _bWeightsGrad = new Tensor<T>([_modelDimension, _modelDimension]);
        _bBiasGrad = new Tensor<T>([_modelDimension]);
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
            var sigmoidDeriv = new Tensor<T>(rGate_t.Shape.ToArray());
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
        var parameters = new Vector<T>(ParameterCount);
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
        var result = new Vector<T>(ParameterCount);
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
        _aWeightsGrad = null;
        _aBiasGrad = null;
        _bWeightsGrad = null;
        _bBiasGrad = null;
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
        _aWeightsGrad = null;
        _aBiasGrad = null;
        _bWeightsGrad = null;
        _bBiasGrad = null;
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

    private Tensor<T>[] GetAllParameterTensors() =>
    [
        _timeMixR, _timeMixK, _timeMixV, _timeMixA, _timeMixB,
        _receptanceWeights, _keyWeights, _valueWeights, _outputWeights,
        _aWeights, _aBias, _bWeights, _bBias,
        _groupNormGamma, _groupNormBeta,
        _channelMixR, _channelMixK,
        _channelKeyWeights, _channelValueWeights, _channelReceptanceWeights,
        _normGamma1, _normBeta1, _normGamma2, _normBeta2
    ];

    private Tensor<T>?[] GetAllGradientTensors() =>
    [
        _timeMixRGrad, _timeMixKGrad, _timeMixVGrad, _timeMixAGrad, _timeMixBGrad,
        _receptanceWeightsGrad, _keyWeightsGrad, _valueWeightsGrad, _outputWeightsGrad,
        _aWeightsGrad, _aBiasGrad, _bWeightsGrad, _bBiasGrad,
        _groupNormGammaGrad, _groupNormBetaGrad,
        _channelMixRGrad, _channelMixKGrad,
        _channelKeyWeightsGrad, _channelValueWeightsGrad, _channelReceptanceWeightsGrad,
        _normGamma1Grad, _normBeta1Grad, _normGamma2Grad, _normBeta2Grad
    ];

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        var xPlaceholder = new Tensor<T>(new int[] { 1, _modelDimension });
        var xNode = TensorOperations<T>.Variable(xPlaceholder, "rwkv7_input");
        var outWeightsNode = TensorOperations<T>.Variable(_outputWeights, "rwkv7_W_out");

        inputNodes.Add(xNode);
        inputNodes.Add(outWeightsNode);

        return TensorOperations<T>.MatrixMultiply(xNode, outWeightsNode);
    }

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

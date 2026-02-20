using AiDotNet.Autodiff;
using AiDotNet.Helpers;

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

    // Channel mixing forward caches (needed for backward pass gradient accumulation)
    private Tensor<T>? _cachedChannelRGate;   // [batch, seqLen, modelDim] sigmoid(r) gate values
    private Tensor<T>? _cachedChannelKSiLU;   // [batch, seqLen, ffnDim] SiLU(k) activation values
    private Tensor<T>? _cachedChannelKProj;   // [batch, seqLen, ffnDim] pre-SiLU k projection values
    private Tensor<T>? _cachedChannelRInput;  // [batch, seqLen, modelDim] token-shifted r input
    private Tensor<T>? _cachedChannelKInput;  // [batch, seqLen, modelDim] token-shifted k input

    // Channel mixing previous token for token shift (persisted across calls)
    private Tensor<T>? _channelMixPrevToken;  // [batch, modelDim]

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
    private Tensor<T>? _recurrentState;  // [batch, numHeads, headDim, headDim]
    private Tensor<T>? _prevToken;       // [batch, modelDim] for token shift

    /// <inheritdoc />
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
        IActivationFunction<T>? activationFunction = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
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
        int fanIn = tensor.Shape[0];
        int fanOut = tensor.Shape[1];
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (fanIn + fanOut)));
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
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

            // Cache for backward
            allR.SetSlice(1, t, r);
            allK.SetSlice(1, t, k);
            allV.SetSlice(1, t, v);
            allA.SetSlice(1, t, aProj);
            allB.SetSlice(1, t, bProj);

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

                            // Read from state: output[di] = sum_vi(S[di, vi] * k[vi])
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

            // Group normalization on WKV output (per head)
            var normedWkv = ApplyGroupNorm(wkvOutput, batchSize);
            allWkv.SetSlice(1, t, normedWkv);

            // Output projection
            var y_t = Engine.TensorMatMul(normedWkv, _outputWeights);
            output.SetSlice(1, t, y_t);

            xPrev = x_t;
        }

        // Store state for autoregressive inference
        _recurrentState = state;
        _prevToken = xPrev;

        // Cache for backward (WKV output used by gradient computation)
        _cachedWkvOut = allWkv;

        return output;
    }

    /// <summary>
    /// Channel mixing forward with SiLU gating (RWKV-7 style).
    /// </summary>
    private Tensor<T> ChannelMixingForward(Tensor<T> x, int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        // Use persisted previous token for token shift (fixes sequential generation)
        var xPrev = _channelMixPrevToken ?? new Tensor<T>(new[] { batchSize, _modelDimension });

        // Allocate caches for backward pass
        _cachedChannelRGate = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        _cachedChannelKSiLU = new Tensor<T>(new[] { batchSize, seqLen, _ffnDimension });
        _cachedChannelKProj = new Tensor<T>(new[] { batchSize, seqLen, _ffnDimension });
        _cachedChannelRInput = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        _cachedChannelKInput = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

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

            // Cache token-shifted inputs for backward
            _cachedChannelRInput.SetSlice(1, t, rInput);
            _cachedChannelKInput.SetSlice(1, t, kInput);

            // r = sigmoid(W_r * rInput)
            var rProj = Engine.TensorMatMul(rInput, _channelReceptanceWeights);
            var rGate = Engine.Sigmoid(rProj);

            // Cache rGate for backward
            _cachedChannelRGate.SetSlice(1, t, rGate);

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

            // Cache kProj and kSiLU for backward
            _cachedChannelKProj.SetSlice(1, t, kProj);
            _cachedChannelKSiLU.SetSlice(1, t, kSiLU);

            // v = W_v * SiLU(k)
            var vProj = Engine.TensorMatMul(kSiLU, _channelValueWeights);  // [batch, modelDim]

            // output = sigmoid(r) * v
            var y_t = Engine.TensorMultiply(rGate, vProj);
            output.SetSlice(1, t, y_t);

            xPrev = x_t;
        }

        // Persist previous token for sequential generation
        _channelMixPrevToken = xPrev;

        return output;
    }

    /// <summary>
    /// Applies group normalization across heads. Each head's dimensions are normalized independently.
    /// </summary>
    private Tensor<T> ApplyGroupNorm(Tensor<T> input, int batchSize)
    {
        var output = new Tensor<T>(input.Shape);
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
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        T epsilon = NumOps.FromDouble(1e-6);

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                T mean = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++)
                    mean = NumOps.Add(mean, input[new[] { bi, t, d }]);
                mean = NumOps.Divide(mean, NumOps.FromDouble(_modelDimension));

                T variance = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++)
                {
                    T diff = NumOps.Subtract(input[new[] { bi, t, d }], mean);
                    variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
                }
                variance = NumOps.Divide(variance, NumOps.FromDouble(_modelDimension));
                T stdInv = NumOps.Divide(NumOps.One, NumOps.Sqrt(NumOps.Add(variance, epsilon)));

                for (int d = 0; d < _modelDimension; d++)
                {
                    T normalized = NumOps.Multiply(
                        NumOps.Subtract(input[new[] { bi, t, d }], mean), stdInv);
                    output[new[] { bi, t, d }] = NumOps.Add(
                        NumOps.Multiply(gamma[d], normalized), beta[d]);
                }
            }
        }

        return output;
    }

    /// <inheritdoc />
    /// <remarks>
    /// This backward pass uses an approximate gradient flow strategy. Gradients flow through
    /// residual connections, and only output projection and receptance weight gradients are
    /// accumulated via <see cref="AccumulateTimeMixGradients"/> and
    /// <see cref="AccumulateChannelMixGradients"/>. The following gradient computations are
    /// NOT implemented (training will be suboptimal):
    /// <list type="bullet">
    ///   <item>Token shift mixing coefficients (_timeMixR/K/V/A/B, _channelMixR/K)</item>
    ///   <item>r/k/v linear projection weights (_receptanceWeights, _keyWeights, _valueWeights)</item>
    ///   <item>Dynamic state evolution weights (_aWeights, _aBias, _bWeights, _bBias)</item>
    ///   <item>Layer normalization parameters (_normGamma1/2, _normBeta1/2)</item>
    ///   <item>Group normalization parameters (_groupNormGamma, _groupNormBeta)</item>
    /// </list>
    /// To extend with full backpropagation through the WKV-7 kernel, implement gradient
    /// accumulation in <see cref="AccumulateTimeMixGradients"/> and
    /// <see cref="AccumulateChannelMixGradients"/> with additional forward-pass caching.
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        grad3D = ApplyActivationDerivative(_lastOutput, grad3D);

        // Initialize gradients
        InitializeGradients();

        // Channel mixing backward (approximate: pass gradient through residual)
        var dAfterTimeMix = grad3D;  // gradient flows through residual
        if (_lastNormed2 != null && _lastChannelMixOutput != null)
        {
            // Accumulate channel mixing weight gradients
            AccumulateChannelMixGradients(grad3D, _lastNormed2, batchSize, seqLen);
            // Gradient through residual: dAfterTimeMix = dOutput (residual pass-through)
        }

        // Time mixing backward (approximate: pass gradient through residual)
        var dInput = dAfterTimeMix;  // gradient flows through residual
        if (_lastNormed1 != null && _lastTimeMixOutput != null)
        {
            // Accumulate time mixing weight gradients
            AccumulateTimeMixGradients(dAfterTimeMix, _lastNormed1, batchSize, seqLen);
        }

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return dInput.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return dInput.Reshape(_originalInputShape);

        return dInput;
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
    /// Computes gradients for the three projection weight matrices (_channelReceptanceWeights,
    /// _channelKeyWeights, _channelValueWeights) using cached forward-pass intermediates.
    /// The channel mixing forward pass is: output = sigmoid(rInput * W_r) * (SiLU(kInput * W_k) * W_v).
    /// </remarks>
    private void AccumulateChannelMixGradients(Tensor<T> dOutput, Tensor<T> normedInput,
        int batchSize, int seqLen)
    {
        if (_channelReceptanceWeightsGrad == null || _channelValueWeightsGrad == null ||
            _channelKeyWeightsGrad == null) return;

        if (_cachedChannelRGate == null || _cachedChannelKSiLU == null ||
            _cachedChannelKProj == null || _cachedChannelRInput == null ||
            _cachedChannelKInput == null) return;

        for (int t = 0; t < seqLen; t++)
        {
            var dOut_t = dOutput.GetSliceAlongDimension(t, 1);  // [batch, modelDim]
            var rGate_t = _cachedChannelRGate.GetSliceAlongDimension(t, 1);  // [batch, modelDim]
            var kSiLU_t = _cachedChannelKSiLU.GetSliceAlongDimension(t, 1);  // [batch, ffnDim]
            var kProj_t = _cachedChannelKProj.GetSliceAlongDimension(t, 1);  // [batch, ffnDim]
            var rInput_t = _cachedChannelRInput.GetSliceAlongDimension(t, 1);  // [batch, modelDim]
            var kInput_t = _cachedChannelKInput.GetSliceAlongDimension(t, 1);  // [batch, modelDim]

            // output = rGate * vProj, where vProj = kSiLU * W_v
            // dVProj = rGate * dOutput (element-wise)
            var dVProj = Engine.TensorMultiply(rGate_t, dOut_t);  // [batch, modelDim]

            // vProj = kSiLU * W_v => dW_v += kSiLU^T * dVProj
            var dWv = Engine.TensorMatMul(kSiLU_t.Transpose(new[] { 1, 0 }), dVProj);
            for (int i = 0; i < _ffnDimension; i++)
                for (int j = 0; j < _modelDimension; j++)
                    _channelValueWeightsGrad[new[] { i, j }] = NumOps.Add(
                        _channelValueWeightsGrad[new[] { i, j }], dWv[new[] { i, j }]);

            // dKSiLU = dVProj * W_v^T
            var dKSiLU = Engine.TensorMatMul(dVProj, _channelValueWeights.Transpose(new[] { 1, 0 }));  // [batch, ffnDim]

            // SiLU derivative: d/dx[x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            //                                        = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            var dKProj = new Tensor<T>(new[] { batchSize, _ffnDimension });
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _ffnDimension; d++)
                {
                    double val = NumOps.ToDouble(kProj_t[new[] { bi, d }]);
                    double sig = 1.0 / (1.0 + Math.Exp(-val));
                    double siluDeriv = sig * (1.0 + val * (1.0 - sig));
                    dKProj[new[] { bi, d }] = NumOps.Multiply(
                        dKSiLU[new[] { bi, d }], NumOps.FromDouble(siluDeriv));
                }
            }

            // kProj = kInput * W_k => dW_k += kInput^T * dKProj
            var dWk = Engine.TensorMatMul(kInput_t.Transpose(new[] { 1, 0 }), dKProj);
            for (int i = 0; i < _modelDimension; i++)
                for (int j = 0; j < _ffnDimension; j++)
                    _channelKeyWeightsGrad[new[] { i, j }] = NumOps.Add(
                        _channelKeyWeightsGrad[new[] { i, j }], dWk[new[] { i, j }]);

            // For receptance: output = rGate * vProj
            // dRGate = dOutput * vProj (element-wise)
            var vProj_t = Engine.TensorMatMul(kSiLU_t, _channelValueWeights);
            var dRGate = Engine.TensorMultiply(dOut_t, vProj_t);  // [batch, modelDim]

            // rGate = sigmoid(rProj), sigmoid derivative: sig * (1 - sig)
            var dRProj = new Tensor<T>(new[] { batchSize, _modelDimension });
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _modelDimension; d++)
                {
                    double rVal = NumOps.ToDouble(rGate_t[new[] { bi, d }]);
                    double sigDeriv = rVal * (1.0 - rVal);
                    dRProj[new[] { bi, d }] = NumOps.Multiply(
                        dRGate[new[] { bi, d }], NumOps.FromDouble(sigDeriv));
                }
            }

            // rProj = rInput * W_r => dW_r += rInput^T * dRProj
            var dWr = Engine.TensorMatMul(rInput_t.Transpose(new[] { 1, 0 }), dRProj);
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
        _cachedChannelKSiLU = null;
        _cachedChannelKProj = null;
        _cachedChannelRInput = null;
        _cachedChannelKInput = null;
        _recurrentState = null;
        _prevToken = null;
        _channelMixPrevToken = null;
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
    /// <remarks>
    /// This is a simplified computation graph for export purposes. The full RWKV-7 computation
    /// (WKV-7 recurrent kernel, token shift, dynamic state evolution via a/b projections,
    /// channel mixing with SiLU gating, layer normalization, group normalization) is not
    /// representable in the current static computation graph format due to recurrent state
    /// dependencies. Only the time-mixing output projection and residual connections are exported.
    /// Omitted components: time mixing (WKV-7 kernel, r/k/v/a/b projections, token shift),
    /// channel mixing (SiLU gating, key/value/receptance projections), layer/group normalization.
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));
        if (inputNodes.Count == 0)
            throw new ArgumentException("At least one input node is required.", nameof(inputNodes));

        // Use the provided input node (from the parent language model's current embedding)
        var xNode = inputNodes[0];
        var outWeightsNode = TensorOperations<T>.Variable(_outputWeights, "rwkv7_W_out");

        inputNodes.Add(outWeightsNode);

        // Time mixing sub-layer (simplified: only output projection)
        var timeMixOut = TensorOperations<T>.MatrixMultiply(xNode, outWeightsNode);
        // Residual connection for time mixing
        var afterTimeMix = TensorOperations<T>.Add(xNode, timeMixOut);

        // Channel mixing sub-layer (simplified: receptance-gated value projection)
        var chRecWeightsNode = TensorOperations<T>.Variable(_channelReceptanceWeights, "rwkv7_W_cr");
        var chKeyWeightsNode = TensorOperations<T>.Variable(_channelKeyWeights, "rwkv7_W_ck");
        var chValWeightsNode = TensorOperations<T>.Variable(_channelValueWeights, "rwkv7_W_cv");
        inputNodes.Add(chRecWeightsNode);
        inputNodes.Add(chKeyWeightsNode);
        inputNodes.Add(chValWeightsNode);

        var rProj = TensorOperations<T>.MatrixMultiply(afterTimeMix, chRecWeightsNode);
        var kProj = TensorOperations<T>.MatrixMultiply(afterTimeMix, chKeyWeightsNode);
        var vProj = TensorOperations<T>.MatrixMultiply(kProj, chValWeightsNode);
        var channelOut = TensorOperations<T>.ElementwiseMultiply(rProj, vProj);

        // Residual connection for channel mixing
        return TensorOperations<T>.Add(afterTimeMix, channelOut);
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

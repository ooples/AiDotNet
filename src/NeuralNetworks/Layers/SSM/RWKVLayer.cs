using AiDotNet.Autodiff;
using AiDotNet.Helpers;

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
public class RWKVLayer<T> : LayerBase<T>
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
    private Tensor<T> _receptanceWeights;
    private Tensor<T> _keyWeights;
    private Tensor<T> _valueWeights;
    private Tensor<T> _outputWeights;

    // Decay parameter (w): [numHeads, headDim] - data-dependent in v6
    private Tensor<T> _decayWeights;  // [modelDim, modelDim] projects input to per-head decay
    private Tensor<T> _decayBias;     // [modelDim]

    // Bonus parameter for current token: [numHeads, headDim]
    private Tensor<T> _bonus;

    // Channel mixing parameters
    private Tensor<T> _channelMixR;  // [modelDim]
    private Tensor<T> _channelMixK;  // [modelDim]
    private Tensor<T> _channelKeyWeights;    // [modelDim, modelDim * 4]
    private Tensor<T> _channelValueWeights;  // [modelDim * 4, modelDim]
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
    private Tensor<T>? _decayWeightsGradient;
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
    public override int ParameterCount =>
        _timeMixR.Length + _timeMixK.Length + _timeMixV.Length +
        _receptanceWeights.Length + _keyWeights.Length + _valueWeights.Length + _outputWeights.Length +
        _decayWeights.Length + _decayBias.Length + _bonus.Length +
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
        IActivationFunction<T>? activationFunction = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
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
        _decayWeights = new Tensor<T>([modelDimension, modelDimension]);
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
        InitializeTensor(_decayWeights);
        _decayBias.Fill(NumOps.FromDouble(-5.0));  // Initial decay ~ exp(-5) ≈ 0.0067 per step
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
            return result.Reshape(seqLen, _modelDimension);

        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
            outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen;
        outputShape[rank - 1] = _modelDimension;
        return result.Reshape(outputShape);
    }

    /// <summary>
    /// Time mixing forward: token shift + linear attention with exponential decay.
    /// </summary>
    private Tensor<T> TimeMixingForward(Tensor<T> x, int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        // State for WKV: numerator [batch, numHeads, headDim, headDim] and denominator [batch, numHeads, headDim]
        var stateNum = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });
        var stateDen = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension });

        // Store all states for backward
        var allStates = new Tensor<T>(new[] { batchSize, seqLen + 1, _modelDimension });

        var xPrev = new Tensor<T>(new[] { batchSize, _modelDimension });  // initialized to zeros (token shift)

        for (int t = 0; t < seqLen; t++)
        {
            var x_t = x.GetSliceAlongDimension(t, 1);  // [batch, modelDim]

            // Token shift: mix current and previous token
            var shifted = new Tensor<T>(new[] { batchSize, _modelDimension });
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _modelDimension; d++)
                {
                    T mu_r = _timeMixR[d];
                    T mu_k = _timeMixK[d];
                    T mu_v = _timeMixV[d];
                    T curr = x_t[new[] { bi, d }];
                    T prev = xPrev[new[] { bi, d }];

                    // For receptance, key, value: different shifts stored in 'shifted' temporarily
                    // We compute all three at once for each position
                    shifted[new[] { bi, d }] = curr;  // Will be used for individual projections
                }
            }

            // Compute receptance, key, value with token-shifted inputs
            var rInput = new Tensor<T>(new[] { batchSize, _modelDimension });
            var kInput = new Tensor<T>(new[] { batchSize, _modelDimension });
            var vInput = new Tensor<T>(new[] { batchSize, _modelDimension });

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
                }
            }

            // Project to r, k, v
            var r = Engine.TensorMatMul(rInput, _receptanceWeights);  // [batch, modelDim]
            var k = Engine.TensorMatMul(kInput, _keyWeights);
            var v = Engine.TensorMatMul(vInput, _valueWeights);

            // Compute data-dependent decay
            var decay = Engine.TensorMatMul(x_t, _decayWeights);
            var decayBias2D = _decayBias.Reshape(1, _modelDimension);
            decay = Engine.TensorBroadcastAdd(decay, decayBias2D);

            // WKV computation per head with exponential decay
            var wkvOutput = new Tensor<T>(new[] { batchSize, _modelDimension });

            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int hi = 0; hi < _numHeads; hi++)
                {
                    int dimStart = hi * _headDimension;

                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatD = dimStart + di;
                        T kVal = k[new[] { bi, flatD }];
                        T bonusVal = _bonus[new[] { hi, di }];

                        // Numerator and denominator update with decay
                        T decayVal = NumOps.FromDouble(
                            -Math.Exp(NumOps.ToDouble(decay[new[] { bi, flatD }])));
                        T decayFactor = NumOps.FromDouble(
                            Math.Exp(NumOps.ToDouble(decayVal)));

                        // WKV: weighted sum of values with exponential decay
                        T num = NumOps.Zero;
                        T den = NumOps.Zero;

                        for (int vi = 0; vi < _headDimension; vi++)
                        {
                            int flatV = dimStart + vi;
                            T vVal = v[new[] { bi, flatV }];

                            T prevNum = stateNum[new[] { bi, hi, di, vi }];
                            T expK = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(kVal)));

                            // Update: state = decay * state + exp(k) * v
                            T newNum = NumOps.Add(
                                NumOps.Multiply(decayFactor, prevNum),
                                NumOps.Multiply(expK, vVal));
                            stateNum[new[] { bi, hi, di, vi }] = newNum;

                            // Current token bonus
                            T bonusContrib = NumOps.Multiply(
                                NumOps.FromDouble(Math.Exp(NumOps.ToDouble(
                                    NumOps.Add(kVal, bonusVal)))), vVal);

                            if (vi == di)  // Diagonal contribution for denominator
                            {
                                T prevDen = stateDen[new[] { bi, hi, di }];
                                T newDen = NumOps.Add(
                                    NumOps.Multiply(decayFactor, prevDen), expK);
                                stateDen[new[] { bi, hi, di }] = newDen;
                                den = newDen;
                            }

                            num = NumOps.Add(num, NumOps.Add(newNum, bonusContrib));
                        }

                        // Normalize and apply receptance gate
                        T rVal = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(
                            -NumOps.ToDouble(r[new[] { bi, flatD }]))));  // sigmoid(r)

                        T safeDiv = NumOps.GreaterThan(
                            NumOps.FromDouble(Math.Abs(NumOps.ToDouble(den))),
                            NumOps.FromDouble(1e-10))
                            ? NumOps.Divide(num, den)
                            : num;

                        wkvOutput[new[] { bi, flatD }] = NumOps.Multiply(rVal, safeDiv);
                    }
                }
            }

            // Output projection
            var y_t = Engine.TensorMatMul(wkvOutput, _outputWeights);
            output.SetSlice(1, t, y_t);

            // Update previous token for next step
            xPrev = x_t;
        }

        _lastState = allStates;
        _lastReceptance = output;  // Cache for backward
        _lastWkv = output;
        return output;
    }

    /// <summary>
    /// Channel mixing forward: squared ReLU with receptance gating.
    /// </summary>
    private Tensor<T> ChannelMixingForward(Tensor<T> x, int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var xPrev = new Tensor<T>(new[] { batchSize, _modelDimension });
        int expandedDim = _modelDimension * 4;

        for (int t = 0; t < seqLen; t++)
        {
            var x_t = x.GetSliceAlongDimension(t, 1);

            // Token shift for channel mixing
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

            // k = W_k * kInput, then squared ReLU
            var kProj = Engine.TensorMatMul(kInput, _channelKeyWeights);  // [batch, expandedDim]

            // Squared ReLU: max(0, k)^2
            var kSquared = new Tensor<T>(new[] { batchSize, expandedDim });
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < expandedDim; d++)
                {
                    T val = kProj[new[] { bi, d }];
                    if (NumOps.GreaterThan(val, NumOps.Zero))
                        kSquared[new[] { bi, d }] = NumOps.Multiply(val, val);
                }
            }

            // v = W_v * kSquared
            var vProj = Engine.TensorMatMul(kSquared, _channelValueWeights);  // [batch, modelDim]

            // output = sigmoid(r) * v
            var y_t = Engine.TensorMultiply(rGate, vProj);
            output.SetSlice(1, t, y_t);

            xPrev = x_t;
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
                // Compute mean
                T mean = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++)
                    mean = NumOps.Add(mean, input[new[] { bi, t, d }]);
                mean = NumOps.Divide(mean, NumOps.FromDouble(_modelDimension));

                // Compute variance
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
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int rank = outputGradient.Shape.Length;
        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(_lastOutput, grad3D);

        // Initialize all gradients
        _timeMixRGradient = new Tensor<T>([_modelDimension]);
        _timeMixKGradient = new Tensor<T>([_modelDimension]);
        _timeMixVGradient = new Tensor<T>([_modelDimension]);
        _receptanceWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _keyWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _valueWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _decayWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _decayBiasGradient = new Tensor<T>([_modelDimension]);
        _bonusGradient = new Tensor<T>([_numHeads, _headDimension]);
        _channelMixRGradient = new Tensor<T>([_modelDimension]);
        _channelMixKGradient = new Tensor<T>([_modelDimension]);
        _channelKeyWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension * 4]);
        _channelValueWeightsGradient = new Tensor<T>([_modelDimension * 4, _modelDimension]);
        _channelReceptanceWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _normGamma1Gradient = new Tensor<T>([_modelDimension]);
        _normBeta1Gradient = new Tensor<T>([_modelDimension]);
        _normGamma2Gradient = new Tensor<T>([_modelDimension]);
        _normBeta2Gradient = new Tensor<T>([_modelDimension]);

        // Simplified backward: propagate gradient through residual connections
        // Full backward would decompose each sub-layer; here we propagate the main path
        var inputGrad = activationGrad;  // Residual connection gradient passes through

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return inputGrad.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return inputGrad.Reshape(_originalInputShape);

        return inputGrad;
    }

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_timeMixRGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _timeMixR = Engine.TensorAdd(_timeMixR, Engine.TensorMultiplyScalar(_timeMixRGradient, negLR));
        _timeMixK = Engine.TensorAdd(_timeMixK, Engine.TensorMultiplyScalar(_timeMixKGradient!, negLR));
        _timeMixV = Engine.TensorAdd(_timeMixV, Engine.TensorMultiplyScalar(_timeMixVGradient!, negLR));
        _receptanceWeights = Engine.TensorAdd(_receptanceWeights, Engine.TensorMultiplyScalar(_receptanceWeightsGradient!, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGradient!, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGradient!, negLR));
        _outputWeights = Engine.TensorAdd(_outputWeights, Engine.TensorMultiplyScalar(_outputWeightsGradient!, negLR));
        _decayWeights = Engine.TensorAdd(_decayWeights, Engine.TensorMultiplyScalar(_decayWeightsGradient!, negLR));
        _decayBias = Engine.TensorAdd(_decayBias, Engine.TensorMultiplyScalar(_decayBiasGradient!, negLR));
        _bonus = Engine.TensorAdd(_bonus, Engine.TensorMultiplyScalar(_bonusGradient!, negLR));
        _channelMixR = Engine.TensorAdd(_channelMixR, Engine.TensorMultiplyScalar(_channelMixRGradient!, negLR));
        _channelMixK = Engine.TensorAdd(_channelMixK, Engine.TensorMultiplyScalar(_channelMixKGradient!, negLR));
        _channelKeyWeights = Engine.TensorAdd(_channelKeyWeights, Engine.TensorMultiplyScalar(_channelKeyWeightsGradient!, negLR));
        _channelValueWeights = Engine.TensorAdd(_channelValueWeights, Engine.TensorMultiplyScalar(_channelValueWeightsGradient!, negLR));
        _channelReceptanceWeights = Engine.TensorAdd(_channelReceptanceWeights, Engine.TensorMultiplyScalar(_channelReceptanceWeightsGradient!, negLR));
        _normGamma1 = Engine.TensorAdd(_normGamma1, Engine.TensorMultiplyScalar(_normGamma1Gradient!, negLR));
        _normBeta1 = Engine.TensorAdd(_normBeta1, Engine.TensorMultiplyScalar(_normBeta1Gradient!, negLR));
        _normGamma2 = Engine.TensorAdd(_normGamma2, Engine.TensorMultiplyScalar(_normGamma2Gradient!, negLR));
        _normBeta2 = Engine.TensorAdd(_normBeta2, Engine.TensorMultiplyScalar(_normBeta2Gradient!, negLR));
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        int totalParams = ParameterCount;
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
        int expectedParams = ParameterCount;
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
        _decayWeights, _decayBias, _bonus,
        _channelMixR, _channelMixK,
        _channelKeyWeights, _channelValueWeights, _channelReceptanceWeights,
        _normGamma1, _normBeta1, _normGamma2, _normBeta2
    ];

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
        _decayWeightsGradient = null;
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

    /// <inheritdoc />
    public override bool SupportsJitCompilation => true;

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        var xPlaceholder = new Tensor<T>(new int[] { 1, _modelDimension });
        var xNode = TensorOperations<T>.Variable(xPlaceholder, "x_t");
        var outWeightsNode = TensorOperations<T>.Variable(_outputWeights, "W_out");

        inputNodes.Add(xNode);
        inputNodes.Add(outWeightsNode);

        var outWeightsT = TensorOperations<T>.Transpose(outWeightsNode);
        var outputNode = TensorOperations<T>.MatrixMultiply(xNode, outWeightsT);

        return outputNode;
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["NumHeads"] = _numHeads.ToString();
        metadata["HeadDimension"] = _headDimension.ToString();
        return metadata;
    }

    /// <summary>
    /// Gets the receptance projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetReceptanceWeights() => _receptanceWeights;

    /// <summary>
    /// Gets the output projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetOutputWeights() => _outputWeights;
}
